@preconcurrency import CoreML
import Foundation

/// Experimental TDT-side vocabulary rescorer.
///
/// Re-runs the preprocessor + encoder over the audio, replays the TDT
/// decoder LSTM through the prefix transcript to recover decoder state,
/// then scores candidate replacement words by teacher-forcing their TDT
/// tokens against the original phrase's emit frames using the
/// `JointDecisionLogits` model. Decisions are made by comparing the
/// summed token log-probs of the candidate vs. the original tokens at
/// the same frames (with the decoder state stepped forward as we go).
///
/// This is a v1 experiment to evaluate whether scoring against the TDT
/// posterior improves rescoring decisions vs. the existing CTC-based
/// path. It does NOT replace the CTC rescorer; both can run side-by-side.
public final class TdtRescorer {

    private let logger = AppLogger(category: "TdtRescorer")

    private let asrModels: AsrModels
    private let tokenizer: SentencePieceTokenizer
    private let modelInference: TdtModelInference
    private let blankId: Int

    // MARK: - Profiling

    /// Total decoder.predict calls observed since this rescorer was
    /// created. Visible to callers for coarse profiling. Not thread-safe.
    public private(set) var decoderCallCount: Int = 0
    /// Wall time accumulated inside decoder.predict.
    public private(set) var decoderSeconds: Double = 0
    /// Total jointLogits.predict calls.
    public private(set) var jointCallCount: Int = 0
    /// Wall time accumulated inside jointLogits.predict.
    public private(set) var jointSeconds: Double = 0
    /// Wall time accumulated inside encoder/preprocessor runs.
    public private(set) var encoderSeconds: Double = 0
    /// Number of encoder.runEncoder() calls.
    public private(set) var encoderCallCount: Int = 0

    /// Reset profiling counters.
    public func resetProfile() {
        decoderCallCount = 0
        decoderSeconds = 0
        jointCallCount = 0
        jointSeconds = 0
        encoderSeconds = 0
        encoderCallCount = 0
    }

    /// SentencePiece word-boundary marker (`▁`). Kept here for cheap repeated
    /// access; mirrors `ASRConstants.sentencePieceWordBoundary`.
    private let wordBoundary: String = ASRConstants.sentencePieceWordBoundary

    /// Result of evaluating a single candidate replacement at the TDT level.
    public struct Decision: Sendable {
        public let originalPhrase: String
        public let candidate: String
        public let originalScore: Float
        public let candidateScore: Float
        public let shouldReplace: Bool
    }

    public init(asrModels: AsrModels, tokenizerModelURL: URL) throws {
        guard asrModels.jointLogits != nil else {
            throw ASRError.processingFailed(
                "TdtRescorer requires asrModels.jointLogits (JointDecisionLogits.mlmodelc)"
            )
        }
        let data = try Data(contentsOf: tokenizerModelURL)
        self.tokenizer = try SentencePieceTokenizer(modelData: data)
        self.asrModels = asrModels
        self.modelInference = TdtModelInference()
        self.blankId = asrModels.version.blankId
    }

    // MARK: - Public API

    /// Score a candidate replacement against the original phrase using the
    /// real TDT posterior.
    ///
    /// - Parameters:
    ///   - originalPhrase: Phrase as it appears in the transcript (e.g. "pharmacist").
    ///   - candidate: Replacement candidate (e.g. "pharmalgen").
    ///   - tokenTimings: TDT token timings for the full transcript.
    ///   - phraseStartTokenIndex: Index into `tokenTimings` of the first token
    ///     of the original phrase.
    ///   - phraseTokenCount: Number of tokens (within `tokenTimings`) that
    ///     comprise the original phrase.
    ///   - encoderOutput: Encoder activations from the same audio that
    ///     produced `tokenTimings` (returned by `runEncoder(...)` below).
    ///   - encoderSequenceLength: Valid frame count in `encoderOutput`.
    /// - Returns: A `Decision` summarizing the TDT-level comparison.
    public func score(
        originalPhrase: String,
        candidate: String,
        tokenTimings: [TokenTiming],
        phraseStartTokenIndex: Int,
        phraseTokenCount: Int,
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int
    ) throws -> Decision {
        guard let jointLogits = asrModels.jointLogits else {
            throw ASRError.processingFailed("TdtRescorer: jointLogits missing")
        }

        let phraseEnd = phraseStartTokenIndex + phraseTokenCount
        guard
            phraseStartTokenIndex >= 0,
            phraseTokenCount > 0,
            phraseEnd <= tokenTimings.count
        else {
            throw ASRError.processingFailed(
                "TdtRescorer: invalid phrase range \(phraseStartTokenIndex)+\(phraseTokenCount) for \(tokenTimings.count) timings"
            )
        }

        // Tokenize candidate against the TDT v2 vocab. The SentencePiece
        // tokenizer prepends `▁`; we always score against that variant since
        // the original transcript phrase begins at a word boundary (we got
        // here via word-level alignment).
        let candidateTokens = tokenizer.encode(candidate)
        guard !candidateTokens.isEmpty else {
            throw ASRError.processingFailed(
                "TdtRescorer: candidate '\(candidate)' produced no tokens")
        }

        // Original phrase tokens come straight from the decoded transcript.
        let prefixTokens = tokenTimings[..<phraseStartTokenIndex].map { $0.tokenId }
        let originalTokens = tokenTimings[phraseStartTokenIndex..<phraseEnd].map { $0.tokenId }
        let originalFrames = tokenTimings[phraseStartTokenIndex..<phraseEnd].map {
            self.frameIndex(for: $0, encoderSequenceLength: encoderSequenceLength)
        }

        // Build encoder view + reusable buffers
        let encoderView = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderSequenceLength,
            expectedHiddenSize: asrModels.version.encoderHiddenSize
        )
        let encoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: encoderView.hiddenSize), 1], dataType: .float32
        )
        let encoderDestPtr = encoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: encoderView.hiddenSize)
        let encoderDestStride = encoderStep.strides.last?.intValue ?? 1

        let decoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1], dataType: .float32
        )

        // Replay the decoder over the prefix to recover state right before
        // the original phrase begins. Use the SOS prime path: the very first
        // decoder call is seeded with `blankId` (matches NeMo SOS=blank
        // convention).
        var state = TdtDecoderState.make(decoderLayers: asrModels.version.decoderLayers)
        let targetArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let targetLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        targetLengthArray[0] = 1

        // Prime with SOS=blank, then replay each prefix token.
        var lastDecoderOutput: MLMultiArray = try seedDecoder(
            state: &state,
            targetArray: targetArray,
            targetLengthArray: targetLengthArray,
            decoderModel: asrModels.decoder
        )
        for token in prefixTokens where !isSpecialToken(token) {
            lastDecoderOutput = try stepDecoder(
                token: token,
                state: &state,
                targetArray: targetArray,
                targetLengthArray: targetLengthArray,
                decoderModel: asrModels.decoder
            )
        }

        // Snapshot state at the phrase start so original/candidate scoring
        // both start from the same point.
        let stateAtPhraseStart = try TdtDecoderState(from: state)

        // Score original tokens at their own emit frames.
        let originalScore = try scoreTokenSequence(
            tokens: originalTokens,
            frames: originalFrames,
            state: stateAtPhraseStart,
            startingDecoderOutput: lastDecoderOutput,
            jointLogits: jointLogits,
            encoderView: encoderView,
            encoderStep: encoderStep,
            encoderDestPtr: encoderDestPtr,
            encoderDestStride: encoderDestStride,
            decoderStep: decoderStep,
            targetArray: targetArray,
            targetLengthArray: targetLengthArray
        )

        // Map candidate tokens onto the original phrase's frame range. When
        // K_cand != K_orig, distribute candidate tokens evenly across the
        // [first..last] original-frame interval (linear interpolation,
        // rounded).
        let candidateFrames = mapCandidateFrames(
            candidateCount: candidateTokens.count,
            originalFrames: originalFrames
        )
        let candidateScore = try scoreTokenSequence(
            tokens: candidateTokens,
            frames: candidateFrames,
            state: stateAtPhraseStart,
            startingDecoderOutput: lastDecoderOutput,
            jointLogits: jointLogits,
            encoderView: encoderView,
            encoderStep: encoderStep,
            encoderDestPtr: encoderDestPtr,
            encoderDestStride: encoderDestStride,
            decoderStep: decoderStep,
            targetArray: targetArray,
            targetLengthArray: targetLengthArray
        )

        let shouldReplace = candidateScore > originalScore
        return Decision(
            originalPhrase: originalPhrase,
            candidate: candidate,
            originalScore: originalScore,
            candidateScore: candidateScore,
            shouldReplace: shouldReplace
        )
    }

    /// Run preprocessor + encoder for the given samples and return the
    /// encoder MLMultiArray + valid frame count. Caller is responsible for
    /// passing the correct number of samples (≤ `ASRConstants.maxModelSamples`).
    public func runEncoder(
        audioSamples: [Float]
    ) async throws -> (encoder: MLMultiArray, sequenceLength: Int) {
        let runStart = Date()
        defer {
            encoderSeconds += Date().timeIntervalSince(runStart)
            encoderCallCount += 1
        }
        let preprocessor = asrModels.preprocessor
        guard let encoder = asrModels.encoder else {
            throw ASRError.processingFailed(
                "TdtRescorer: split encoder model required (fused frontends are not supported in v1)"
            )
        }

        // Build preprocessor input
        let audioArray = try MLMultiArray(
            shape: [1, NSNumber(value: audioSamples.count)],
            dataType: .float32
        )
        audioSamples.withUnsafeBufferPointer { src in
            let dst = audioArray.dataPointer.bindMemory(to: Float.self, capacity: audioSamples.count)
            memcpy(dst, src.baseAddress!, audioSamples.count * MemoryLayout<Float>.stride)
        }
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: audioSamples.count)

        let preInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])
        let preOutput = try await preprocessor.prediction(from: preInput)

        // Build encoder input by forwarding all preprocessor features the
        // encoder expects.
        var encoderInputDict: [String: MLFeatureValue] = [:]
        for name in encoder.modelDescription.inputDescriptionsByName.keys {
            if let value = preOutput.featureValue(for: name) {
                encoderInputDict[name] = value
            } else if let value = preInput.featureValue(for: name) {
                encoderInputDict[name] = value
            } else {
                throw ASRError.processingFailed(
                    "TdtRescorer: encoder input '\(name)' not found in preprocessor output or original input"
                )
            }
        }
        let encInput = try MLDictionaryFeatureProvider(dictionary: encoderInputDict)
        let encOutput = try await encoder.prediction(from: encInput)

        guard let encArray = encOutput.featureValue(for: "encoder")?.multiArrayValue else {
            throw ASRError.processingFailed("TdtRescorer: encoder output missing 'encoder' feature")
        }
        guard let encLen = encOutput.featureValue(for: "encoder_length")?.multiArrayValue else {
            throw ASRError.processingFailed(
                "TdtRescorer: encoder output missing 'encoder_length' feature")
        }
        let sequenceLength = encLen[0].intValue
        return (encArray, sequenceLength)
    }

    // MARK: - Private helpers

    /// Convert a `TokenTiming` to an encoder-frame index, clamped to the
    /// valid range of the encoder output.
    private func frameIndex(
        for timing: TokenTiming, encoderSequenceLength: Int
    ) -> Int {
        let frame = Int((timing.startTime / ASRConstants.secondsPerEncoderFrame).rounded())
        return max(0, min(frame, encoderSequenceLength - 1))
    }

    private func isSpecialToken(_ tokenId: Int) -> Bool {
        return tokenId == blankId
    }

    /// Distribute K_cand tokens across the original phrase's frame interval.
    /// When candidate has 1 token, anchors to the first original frame.
    /// Otherwise linear interpolation across [firstFrame, lastFrame].
    private func mapCandidateFrames(
        candidateCount: Int, originalFrames: [Int]
    ) -> [Int] {
        guard let firstFrame = originalFrames.first else { return [] }
        guard candidateCount > 1, let lastFrame = originalFrames.last,
            lastFrame > firstFrame
        else {
            return Array(repeating: firstFrame, count: candidateCount)
        }
        let span = Double(lastFrame - firstFrame)
        return (0..<candidateCount).map { i in
            let frac = Double(i) / Double(candidateCount - 1)
            return firstFrame + Int((frac * span).rounded())
        }
    }

    /// Run the joint-logits model at one (encoder_frame, decoder_step) pair
    /// and return `log P(actualToken)` for the supplied token. Updates the
    /// reusable encoder/decoder buffers in place.
    private func scoreTokenAtFrame(
        token: Int,
        encoderFrame: Int,
        decoderProjection: MLMultiArray,
        jointLogits: MLModel,
        encoderView: EncoderFrameView,
        encoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        decoderStep: MLMultiArray
    ) throws -> Float {
        try encoderView.copyFrame(
            at: encoderFrame, into: encoderDestPtr, destinationStride: encoderDestStride)

        try modelInference.normalizeDecoderProjection(decoderProjection, into: decoderStep)

        let t0 = Date()
        let (tokenLogits, _) = try modelInference.runJointLogits(
            encoderStep: encoderStep,
            decoderStep: decoderStep,
            model: jointLogits
        )
        jointSeconds += Date().timeIntervalSince(t0)
        jointCallCount += 1
        guard token >= 0, token < tokenLogits.count else {
            throw ASRError.processingFailed(
                "TdtRescorer: token id \(token) out of range for logits of size \(tokenLogits.count)"
            )
        }
        return logSoftmaxValue(tokenLogits, at: token)
    }

    /// Walk a token sequence: at each step score the token at its anchor
    /// frame, then advance the decoder LSTM with that token.
    private func scoreTokenSequence(
        tokens: [Int],
        frames: [Int],
        state initialState: TdtDecoderState,
        startingDecoderOutput: MLMultiArray,
        jointLogits: MLModel,
        encoderView: EncoderFrameView,
        encoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        decoderStep: MLMultiArray,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray
    ) throws -> Float {
        guard tokens.count == frames.count else {
            throw ASRError.processingFailed(
                "TdtRescorer: tokens/frames length mismatch \(tokens.count) vs \(frames.count)"
            )
        }

        var state = try TdtDecoderState(from: initialState)
        var currentDecoderOutput = startingDecoderOutput
        var totalLogProb: Float = 0

        for (idx, token) in tokens.enumerated() {
            // Score this token using the *current* decoder projection
            let logProb = try scoreTokenAtFrame(
                token: token,
                encoderFrame: frames[idx],
                decoderProjection: currentDecoderOutput,
                jointLogits: jointLogits,
                encoderView: encoderView,
                encoderStep: encoderStep,
                encoderDestPtr: encoderDestPtr,
                encoderDestStride: encoderDestStride,
                decoderStep: decoderStep
            )
            totalLogProb += logProb

            // Advance the decoder LSTM with the *just-scored* token. This
            // mirrors the TDT decoding loop where the LSTM state at the
            // next step incorporates the most recently emitted token.
            currentDecoderOutput = try stepDecoder(
                token: token,
                state: &state,
                targetArray: targetArray,
                targetLengthArray: targetLengthArray,
                decoderModel: asrModels.decoder
            )
        }

        return totalLogProb
    }

    /// Seed the decoder with SOS (blank) and return the resulting decoder
    /// projection.
    private func seedDecoder(
        state: inout TdtDecoderState,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray,
        decoderModel: MLModel
    ) throws -> MLMultiArray {
        return try stepDecoder(
            token: blankId,
            state: &state,
            targetArray: targetArray,
            targetLengthArray: targetLengthArray,
            decoderModel: decoderModel
        )
    }

    private func stepDecoder(
        token: Int,
        state: inout TdtDecoderState,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray,
        decoderModel: MLModel
    ) throws -> MLMultiArray {
        let t0 = Date()
        let (output, newState) = try modelInference.runDecoder(
            token: token,
            state: state,
            model: decoderModel,
            targetArray: targetArray,
            targetLengthArray: targetLengthArray
        )
        decoderSeconds += Date().timeIntervalSince(t0)
        decoderCallCount += 1
        state = newState
        guard let projection = output.featureValue(for: "decoder")?.multiArrayValue else {
            throw ASRError.processingFailed(
                "TdtRescorer: decoder output missing 'decoder' projection")
        }
        return projection
    }

    /// Compute log-softmax value at a single index from raw logits. We do
    /// the full denominator each time rather than caching per-row because
    /// each decoder step produces a fresh logit vector.
    private func logSoftmaxValue(_ logits: [Float], at index: Int) -> Float {
        guard !logits.isEmpty else { return -.infinity }
        var maxVal: Float = -.infinity
        for v in logits { if v > maxVal { maxVal = v } }
        var sumExp: Float = 0
        for v in logits { sumExp += expf(v - maxVal) }
        return (logits[index] - maxVal) - logf(sumExp)
    }
}
