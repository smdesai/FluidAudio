import Accelerate
@preconcurrency import CoreML
import Foundation

/// Model inference operations for TDT decoding.
///
/// Encapsulates execution of decoder LSTM, joint network, and decoder projection normalization.
/// These operations are separated from the main decoding loop to improve testability and clarity.
internal struct TdtModelInference: Sendable {
    private let predictionOptions: MLPredictionOptions

    init() {
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
    }

    /// Execute decoder LSTM with state caching.
    ///
    /// - Parameters:
    ///   - token: Token ID to decode
    ///   - state: Current decoder LSTM state
    ///   - model: Decoder MLModel
    ///   - targetArray: Pre-allocated array for token input
    ///   - targetLengthArray: Pre-allocated array for length (always 1)
    ///
    /// - Returns: Tuple of (output features, updated state)
    func runDecoder(
        token: Int,
        state: TdtDecoderState,
        model: MLModel,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray
    ) throws -> (output: MLFeatureProvider, newState: TdtDecoderState) {

        // Reuse pre-allocated arrays
        targetArray[0] = NSNumber(value: token)
        // targetLengthArray[0] is already set to 1 and never changes

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: state.hiddenState),
            "c_in": MLFeatureValue(multiArray: state.cellState),
        ])

        // Reuse decoder state output buffers to avoid CoreML allocating new ones
        // Note: outputBackings expects raw backing objects (MLMultiArray / CVPixelBuffer)
        predictionOptions.outputBackings = [
            "h_out": state.hiddenState,
            "c_out": state.cellState,
        ]

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Execute joint network with zero-copy and ANE optimization.
    ///
    /// - Parameters:
    ///   - encoderFrames: View into encoder output tensor
    ///   - timeIndex: Frame index to process
    ///   - preparedDecoderStep: Normalized decoder projection
    ///   - model: Joint MLModel
    ///   - encoderStep: Pre-allocated encoder step array
    ///   - encoderDestPtr: Pointer for encoder frame copy
    ///   - encoderDestStride: Stride for encoder copy
    ///   - inputProvider: Reusable feature provider
    ///   - tokenIdBacking: Pre-allocated output for token ID
    ///   - tokenProbBacking: Pre-allocated output for probability
    ///   - durationBacking: Pre-allocated output for duration
    ///   - needsTopK: When `true`, extract `top_k_ids` / `top_k_logits` from
    ///     the joint output (JointDecisionv3). Callers should pass `true` only
    ///     when a caller-level feature (e.g. language-aware script filtering)
    ///     actually consumes the top-K; otherwise the K-length Swift arrays
    ///     are allocated per step and thrown away.
    ///
    /// - Returns: Joint decision (token, probability, duration bin)
    func runJointPrepared(
        encoderFrames: EncoderFrameView,
        timeIndex: Int,
        preparedDecoderStep: MLMultiArray,
        model: MLModel,
        encoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        inputProvider: MLFeatureProvider,
        tokenIdBacking: MLMultiArray,
        tokenProbBacking: MLMultiArray,
        durationBacking: MLMultiArray,
        needsTopK: Bool = false
    ) throws -> TdtJointDecision {

        // Fill encoder step with the requested frame
        try encoderFrames.copyFrame(at: timeIndex, into: encoderDestPtr, destinationStride: encoderDestStride)

        // Prefetch arrays for ANE
        encoderStep.prefetchToNeuralEngine()
        preparedDecoderStep.prefetchToNeuralEngine()

        // Reuse tiny output tensors for joint prediction (provide raw MLMultiArray backings)
        predictionOptions.outputBackings = [
            "token_id": tokenIdBacking,
            "token_prob": tokenProbBacking,
            "duration": durationBacking,
        ]

        // Execute joint network using the reusable provider
        let output = try model.prediction(
            from: inputProvider,
            options: predictionOptions
        )

        let tokenIdArray = try extractFeatureValue(
            from: output, key: "token_id", errorMessage: "Joint decision output missing token_id")
        let tokenProbArray = try extractFeatureValue(
            from: output, key: "token_prob", errorMessage: "Joint decision output missing token_prob")
        let durationArray = try extractFeatureValue(
            from: output, key: "duration", errorMessage: "Joint decision output missing duration")

        guard tokenIdArray.count == 1,
            tokenProbArray.count == 1,
            durationArray.count == 1
        else {
            throw ASRError.processingFailed("Joint decision returned unexpected tensor shapes")
        }

        let tokenPointer = tokenIdArray.dataPointer.bindMemory(to: Int32.self, capacity: tokenIdArray.count)
        let token = Int(tokenPointer[0])
        let probPointer = tokenProbArray.dataPointer.bindMemory(to: Float.self, capacity: tokenProbArray.count)
        let probability = probPointer[0]
        let durationPointer = durationArray.dataPointer.bindMemory(to: Int32.self, capacity: durationArray.count)
        let durationBin = Int(durationPointer[0])

        // Extract top-K outputs only when the caller requested them. Skipping
        // this in the common (non-filtering) path saves K-length Swift array
        // allocations per decoded step.
        var topKIds: [Int]? = nil
        var topKLogits: [Float]? = nil
        if needsTopK {
            topKIds = try extractInt32Array(from: output, key: "top_k_ids")
            topKLogits = try extractFloat32Array(from: output, key: "top_k_logits")

            // Enforce that top-K outputs are present as a pair with matching
            // lengths. This catches export-schema drift (e.g. only one of the
            // two keys exposed, or K sizes diverging) before the consumer has
            // to defend against it.
            switch (topKIds, topKLogits) {
            case (nil, nil):
                break
            case (let ids?, let logits?):
                guard ids.count == logits.count else {
                    throw ASRError.processingFailed(
                        "Joint decision top-K length mismatch: \(ids.count) vs \(logits.count)")
                }
            default:
                throw ASRError.processingFailed(
                    "Joint decision top-K outputs must be present as a pair (top_k_ids + top_k_logits)")
            }
        }

        return TdtJointDecision(
            token: token,
            probability: probability,
            durationBin: durationBin,
            topKIds: topKIds,
            topKLogits: topKLogits
        )
    }

    /// Execute the logits-only joint variant for a single (encoder_step,
    /// decoder_step) pair and return the per-step `token_logits` and
    /// `duration_logits` as flat `[Float]` rows.
    ///
    /// The model exposes the full TDT posterior (1025 token slots = 1024 vocab
    /// + 1 blank, plus 5 duration buckets), allowing rescoring against the
    /// real TDT joint without indirection through the parakeet-ctc-110m
    /// model.
    ///
    /// - Parameters:
    ///   - model: Loaded `JointDecisionLogits` MLModel
    ///   - encoderStep: `[1, 1024, 1]` Float32 — same buffer shape as the
    ///     standard joint takes; pass the buffer that the caller already
    ///     populates for the regular joint call to share frame copies.
    ///   - decoderStep: `[1, 640, 1]` Float32 — normalized decoder projection
    ///     from `normalizeDecoderProjection`.
    ///   - tokenLogitsBacking: Optional `[1,1,1,1025]` Float32 buffer to reuse
    ///     for `token_logits`. If `nil`, CoreML allocates per call.
    ///   - durationLogitsBacking: Optional `[1,1,1,5]` Float32 buffer to
    ///     reuse for `duration_logits`.
    ///
    /// - Returns: `(tokenLogits, durationLogits)`. `tokenLogits.count` is
    ///   `1025` (vocab + blank); `durationLogits.count` is `5`.
    func runJointLogits(
        encoderStep: MLMultiArray,
        decoderStep: MLMultiArray,
        model: MLModel,
        tokenLogitsBacking: MLMultiArray? = nil,
        durationLogitsBacking: MLMultiArray? = nil
    ) throws -> (tokenLogits: [Float], durationLogits: [Float]) {
        encoderStep.prefetchToNeuralEngine()
        decoderStep.prefetchToNeuralEngine()

        if let tokenLogitsBacking, let durationLogitsBacking {
            predictionOptions.outputBackings = [
                "token_logits": tokenLogitsBacking,
                "duration_logits": durationLogitsBacking,
            ]
        } else {
            predictionOptions.outputBackings = [:]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: decoderStep),
        ])

        let output = try model.prediction(from: input, options: predictionOptions)

        let tokenLogitsArray = try extractFeatureValue(
            from: output, key: "token_logits",
            errorMessage: "JointDecisionLogits output missing token_logits"
        )
        let durationLogitsArray = try extractFeatureValue(
            from: output, key: "duration_logits",
            errorMessage: "JointDecisionLogits output missing duration_logits"
        )

        guard tokenLogitsArray.dataType == .float32, durationLogitsArray.dataType == .float32 else {
            throw ASRError.processingFailed(
                "JointDecisionLogits output dtype mismatch — expected Float32"
            )
        }

        let tokenCount = tokenLogitsArray.count
        let durationCount = durationLogitsArray.count
        let tokenPtr = tokenLogitsArray.dataPointer.bindMemory(to: Float.self, capacity: tokenCount)
        let durationPtr = durationLogitsArray.dataPointer.bindMemory(
            to: Float.self, capacity: durationCount)

        let tokenLogits = Array(UnsafeBufferPointer(start: tokenPtr, count: tokenCount))
        let durationLogits = Array(UnsafeBufferPointer(start: durationPtr, count: durationCount))
        return (tokenLogits, durationLogits)
    }

    /// Normalize decoder projection into [1, hiddenSize, 1] layout via BLAS copy.
    ///
    /// CoreML decoder outputs can have varying layouts ([1, 1, 640] or [1, 640, 1]).
    /// This function normalizes to the joint network's expected input format using
    /// efficient BLAS operations to handle arbitrary strides.
    ///
    /// - Parameters:
    ///   - projection: Decoder output projection (any 3D layout with hiddenSize dimension)
    ///   - destination: Optional pre-allocated destination array (for hot path)
    ///
    /// - Returns: Normalized array in [1, hiddenSize, 1] format
    @discardableResult
    func normalizeDecoderProjection(
        _ projection: MLMultiArray,
        into destination: MLMultiArray? = nil
    ) throws -> MLMultiArray {
        let hiddenSize = ASRConstants.decoderHiddenSize
        let shape = projection.shape.map { $0.intValue }

        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid decoder projection rank: \(shape)")
        }
        guard shape[0] == 1 else {
            throw ASRError.processingFailed("Unsupported decoder batch dimension: \(shape[0])")
        }
        guard projection.dataType == .float32 || projection.dataType == .float16 else {
            throw ASRError.processingFailed("Unsupported decoder projection type: \(projection.dataType)")
        }

        let hiddenAxis: Int
        if shape[2] == hiddenSize {
            hiddenAxis = 2
        } else if shape[1] == hiddenSize {
            hiddenAxis = 1
        } else {
            throw ASRError.processingFailed("Decoder projection hidden size mismatch: \(shape)")
        }

        let timeAxis = (0...2).first { $0 != hiddenAxis && $0 != 0 } ?? 1
        guard shape[timeAxis] == 1 else {
            throw ASRError.processingFailed("Decoder projection time axis must be 1: \(shape)")
        }

        let out: MLMultiArray
        if let destination {
            let outShape = destination.shape.map { $0.intValue }
            guard destination.dataType == .float32, outShape.count == 3, outShape[0] == 1,
                outShape[2] == 1, outShape[1] == hiddenSize
            else {
                throw ASRError.processingFailed(
                    "Prepared decoder step shape mismatch: \(outShape.map(String.init).joined(separator: "x"))")
            }
            out = destination
        } else {
            out = try ANEMemoryUtils.createAlignedArray(
                shape: [1, NSNumber(value: hiddenSize), 1],
                dataType: .float32
            )
        }

        let strides = projection.strides.map { $0.intValue }
        let hiddenStride = strides[hiddenAxis]

        let destPtr = out.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let destStrides = out.strides.map { $0.intValue }
        let destHiddenStride = destStrides[1]

        if projection.dataType == .float32 {
            let dataPointer = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
            let startPtr = dataPointer.advanced(by: 0)
            let destStrideCblas = try makeBlasIndex(destHiddenStride, label: "Decoder destination stride")
            let count = try makeBlasIndex(hiddenSize, label: "Decoder projection length")
            let stride = try makeBlasIndex(hiddenStride, label: "Decoder projection stride")
            cblas_scopy(count, startPtr, stride, destPtr, destStrideCblas)
            return out
        }

        let dataPointer = projection.dataPointer.bindMemory(to: UInt16.self, capacity: projection.count)
        if hiddenStride == 1 && destHiddenStride == 1 {
            var src = vImage_Buffer(
                data: dataPointer,
                height: 1,
                width: vImagePixelCount(hiddenSize),
                rowBytes: hiddenSize * MemoryLayout<UInt16>.stride
            )
            var dst = vImage_Buffer(
                data: destPtr,
                height: 1,
                width: vImagePixelCount(hiddenSize),
                rowBytes: hiddenSize * MemoryLayout<Float>.stride
            )
            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        } else {
            var packed = [UInt16](repeating: 0, count: hiddenSize)
            for hiddenIndex in 0..<hiddenSize {
                packed[hiddenIndex] = dataPointer[hiddenIndex * hiddenStride]
            }
            var converted = [Float](repeating: 0, count: hiddenSize)
            packed.withUnsafeBufferPointer { packedBuffer in
                converted.withUnsafeMutableBufferPointer { convertedBuffer in
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: packedBuffer.baseAddress!),
                        height: 1,
                        width: vImagePixelCount(hiddenSize),
                        rowBytes: hiddenSize * MemoryLayout<UInt16>.stride
                    )
                    var dst = vImage_Buffer(
                        data: convertedBuffer.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(hiddenSize),
                        rowBytes: hiddenSize * MemoryLayout<Float>.stride
                    )
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            for hiddenIndex in 0..<hiddenSize {
                destPtr[hiddenIndex * destHiddenStride] = converted[hiddenIndex]
            }
        }

        return out
    }

    /// Extract MLMultiArray feature value with error handling.
    private func extractFeatureValue(
        from output: MLFeatureProvider, key: String, errorMessage: String
    ) throws
        -> MLMultiArray
    {
        guard let value = output.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }

    /// Read a 1D Int32 feature into an `[Int]`, or `nil` if the feature is absent.
    /// Validates dtype; the 1D contiguous-layout assumption matches the scalar
    /// extraction paths above (CoreML 1D outputs are row-major contiguous).
    private func extractInt32Array(from output: MLFeatureProvider, key: String) throws -> [Int]? {
        guard let array = output.featureValue(for: key)?.multiArrayValue else { return nil }
        guard array.dataType == .int32 else {
            throw ASRError.processingFailed("Expected Int32 for \(key), got \(array.dataType.rawValue)")
        }
        let count = array.count
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
        return (0..<count).map { Int(pointer[$0]) }
    }

    /// Read a 1D Float32 feature into a `[Float]`, or `nil` if the feature is absent.
    private func extractFloat32Array(from output: MLFeatureProvider, key: String) throws -> [Float]? {
        guard let array = output.featureValue(for: key)?.multiArrayValue else { return nil }
        guard array.dataType == .float32 else {
            throw ASRError.processingFailed("Expected Float32 for \(key), got \(array.dataType.rawValue)")
        }
        let count = array.count
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }
}
