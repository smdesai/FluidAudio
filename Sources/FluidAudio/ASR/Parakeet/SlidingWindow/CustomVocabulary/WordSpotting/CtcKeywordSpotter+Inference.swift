import Accelerate
import CoreML
import Foundation

/// CoreML inference pipeline for CTC keyword spotting.
///
/// Handles audio preprocessing, mel spectrogram computation, encoder inference,
/// and log-probability extraction from CTC model outputs.
extension CtcKeywordSpotter {

    // MARK: - Log Probability Computation

    func computeLogProbs(for audioSamples: [Float]) async throws -> CtcLogProbResult {
        guard !audioSamples.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }

        // For audio longer than model limit, use chunked processing
        if audioSamples.count > maxModelSamples {
            return try await computeLogProbsChunked(audioSamples: audioSamples)
        }

        // Use staged models (mel spectrogram + encoder) for short audio
        return try await computeWithStagedModels(audioSamples: audioSamples)
    }

    /// Process long audio in chunks with overlap, concatenating log-probs.
    ///
    /// Algorithm:
    /// 1. Split audio into chunks of maxModelSamples with chunkOverlapSamples overlap
    /// 2. Run CTC inference on each chunk
    /// 3. Concatenate log-probs, averaging overlapping frames
    private func computeLogProbsChunked(audioSamples: [Float]) async throws -> CtcLogProbResult {
        let totalSamples = audioSamples.count
        let chunkSize = maxModelSamples
        let overlap = chunkOverlapSamples
        let stride = chunkSize - overlap

        // Calculate number of chunks needed
        var chunks: [(start: Int, end: Int)] = []
        var start = 0
        while start < totalSamples {
            let end = min(start + chunkSize, totalSamples)
            chunks.append((start: start, end: end))
            if end >= totalSamples { break }
            start += stride
        }

        if debugMode {
            logger.debug("=== Chunked CTC Processing ===")
            logger.debug(
                "Total samples: \(totalSamples) (\(String(format: "%.2f", Double(totalSamples) / Double(sampleRate)))s)"
            )
            logger.debug("Chunk size: \(chunkSize), overlap: \(overlap), stride: \(stride)")
            logger.debug("Number of chunks: \(chunks.count)")
        }

        // Process each chunk
        var chunkResults: [CtcLogProbResult] = []
        for (idx, chunk) in chunks.enumerated() {
            let chunkAudio = Array(audioSamples[chunk.start..<chunk.end])

            if debugMode {
                let startTime = Double(chunk.start) / Double(sampleRate)
                let endTime = Double(chunk.end) / Double(sampleRate)
                logger.debug(
                    "  Chunk \(idx + 1)/\(chunks.count): samples [\(chunk.start)-\(chunk.end)] = [\(String(format: "%.2f", startTime))-\(String(format: "%.2f", endTime))s]"
                )
            }

            let result = try await computeWithStagedModels(audioSamples: chunkAudio)
            chunkResults.append(result)

            if debugMode {
                logger.debug(
                    "    -> \(result.totalFrames) frames, frameDuration=\(String(format: "%.4f", result.frameDuration))s"
                )
            }
        }

        guard !chunkResults.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }

        // Use frame duration from first chunk (should be consistent)
        let frameDuration = chunkResults[0].frameDuration
        guard frameDuration > 0 else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0)
        }

        // Calculate overlap in frames
        let overlapFrames = Int(Double(overlap) / Double(sampleRate) / frameDuration)

        // Concatenate log-probs with overlap averaging
        var concatenatedLogProbs: [[Float]] = []

        for (chunkIdx, result) in chunkResults.enumerated() {
            let logProbs = result.logProbs
            guard !logProbs.isEmpty else { continue }

            if chunkIdx == 0 {
                // First chunk: take all frames
                concatenatedLogProbs.append(contentsOf: logProbs)
            } else {
                // Subsequent chunks: average overlap region, then append non-overlapping part
                let overlapCount = min(overlapFrames, concatenatedLogProbs.count, logProbs.count)

                if overlapCount > 0 {
                    let existingStart = concatenatedLogProbs.count - overlapCount
                    for i in 0..<overlapCount {
                        let existingIdx = existingStart + i
                        concatenatedLogProbs[existingIdx] = Self.mergeOverlapFrame(
                            existing: concatenatedLogProbs[existingIdx],
                            incoming: logProbs[i]
                        )
                    }
                }

                // Append non-overlapping frames from this chunk
                if overlapCount < logProbs.count {
                    concatenatedLogProbs.append(contentsOf: logProbs.suffix(from: overlapCount))
                }
            }
        }

        if debugMode {
            logger.debug("Concatenated: \(concatenatedLogProbs.count) total frames")
            logger.debug("Overlap frames averaged: \(overlapFrames) per boundary")
            logger.debug("==============================")
        }

        return CtcLogProbResult(
            logProbs: concatenatedLogProbs,
            frameDuration: frameDuration,
            totalFrames: concatenatedLogProbs.count,
            audioSamplesUsed: totalSamples
        )
    }

    // MARK: - Staged Model Inference

    private func computeWithStagedModels(audioSamples: [Float]) async throws -> CtcLogProbResult {
        let ctcProfileEnabled =
            ProcessInfo.processInfo.environment["FA_CTC_PROFILE"] != nil

        // Prepare fixed-length audio input expected by MelSpectrogram.
        let (audioInput, clampedCount) = try prepareAudioArray(audioSamples)
        let melInput = try makeAudioFeatureProvider(array: audioInput, length: clampedCount)

        let melModel = models.melSpectrogram
        let encoderModel = models.encoder

        let melT0 = Date()
        let melOutput = try await melModel.compatPrediction(
            from: melInput,
            options: predictionOptions
        )
        let melDt = Date().timeIntervalSince(melT0)

        guard let melFeatures = melOutput.featureValue(for: "melspectrogram_features")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing melspectrogram_features from CTC MelSpectrogram model")
        }

        // Prefer explicit mel_length; otherwise infer from shape (frames axis).
        var melLengthValue =
            melOutput.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue
            ?? melFeatures.shape.last?.intValue
        if melFeatures.shape.count == 4 {
            melLengthValue = melFeatures.shape[2].intValue
        }

        if debugMode {
            logger.debug(
                "Mel features shape: \(melFeatures.shape), mel_length: \(melLengthValue.map(String.init) ?? "nil")")
        }

        // Build encoder input (mel features + length placeholder).
        let encoderInput = try makeEncoderInput(melFeatures: melFeatures, melLength: melLengthValue)

        // Run AudioEncoder to obtain CTC logits.
        let encT0 = Date()
        let encoderOutput = try await encoderModel.compatPrediction(
            from: encoderInput,
            options: predictionOptions
        )
        let encDt = Date().timeIntervalSince(encT0)

        // Check which output is available
        let hasRaw = encoderOutput.featureValue(for: "ctc_head_raw_output")?.multiArrayValue != nil
        let hasSoftmax = encoderOutput.featureValue(for: "ctc_head_output")?.multiArrayValue != nil

        if debugMode {
            logger.debug("CTC outputs available: ctc_head_raw_output=\(hasRaw), ctc_head_output=\(hasSoftmax)")
        }

        // Use ctc_head_raw_output (raw logits), NOT ctc_head_output (which contains post-softmax probabilities)
        // From debugging: ctc_head_output produces nonsense scores when passed through log-softmax again
        let ctcRaw =
            encoderOutput.featureValue(for: "ctc_head_raw_output")?.multiArrayValue
            ?? encoderOutput.featureValue(for: "ctc_head_output")?.multiArrayValue

        guard let ctcRaw else {
            throw ASRError.processingFailed(
                "Missing CTC head output from encoder model (expected ctc_head_raw_output or ctc_head_output)"
            )
        }

        if debugMode {
            logger.debug("CTC raw output shape: \(ctcRaw.shape)")
            let usedOutput = hasRaw ? "ctc_head_raw_output (raw logits)" : "ctc_head_output (post-softmax)"
            logger.debug("Using output: \(usedOutput)")
        }

        // Convert logits -> log-probabilities and trim padding frames.
        // Apply temperature scaling (CTC_TEMPERATURE) and blank bias (BLANK_BIAS)
        let postT0 = Date()
        let allLogProbs = try makeLogProbs(from: ctcRaw, temperature: temperature, blankBias: blankBias)
        let trimmed = trimLogProbs(allLogProbs, audioSampleCount: clampedCount)
        let frameCount = trimmed.count
        let postDt = Date().timeIntervalSince(postT0)

        if ctcProfileEnabled {
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "  CTC-PROFILE-INNER mel=%.3fs encoder=%.3fs post=%.3fs (%d frames, vocab=%d)\n",
                        melDt, encDt, postDt, frameCount, trimmed.first?.count ?? 0
                    ).utf8))
        }

        if debugMode {
            logger.debug(
                "Log-probs: \(trimmed.count) frames (total: \(allLogProbs.count)), vocab size: \(trimmed.first?.count ?? 0)"
            )
        }

        let frameDuration =
            frameCount > 0
            ? Double(clampedCount) / Double(frameCount) / Double(sampleRate)
            : 0

        return CtcLogProbResult(
            logProbs: trimmed,
            frameDuration: frameDuration,
            totalFrames: frameCount,
            audioSamplesUsed: clampedCount
        )
    }

    // MARK: - Audio Preparation

    private func prepareAudioArray(_ audioSamples: [Float]) throws -> (MLMultiArray, Int) {
        let clampedCount = min(audioSamples.count, maxModelSamples)

        // Detect expected input rank from the MelSpectrogram model's 'audio' feature description.
        // Canary-1b-v2 expects rank 1 [samples], parakeet-ctc-0.6b expects rank 2 [1, samples].
        let melModel = models.melSpectrogram
        let audioDesc = melModel.modelDescription.inputDescriptionsByName["audio"]
        let expectedRank = audioDesc?.multiArrayConstraint?.shape.count ?? 1

        // Determine data type - prefer float16 if model expects it, otherwise float32
        let dataType: MLMultiArrayDataType =
            audioDesc?.multiArrayConstraint?.dataType == .float16 ? .float16 : .float32

        let array: MLMultiArray
        if expectedRank == 2 {
            // Rank 2: [1, maxSamples]
            array = try MLMultiArray(shape: [1, NSNumber(value: maxModelSamples)], dataType: dataType)
        } else {
            // Rank 1: [maxSamples]
            array = try MLMultiArray(shape: [NSNumber(value: maxModelSamples)], dataType: dataType)
        }

        // Copy actual samples (MLMultiArray is zero-initialized, so padding is implicit).
        for i in 0..<clampedCount {
            array[i] = NSNumber(value: audioSamples[i])
        }

        if debugMode {
            let midpoint = clampedCount / 2
            var sampleVals: [String] = []
            for i in midpoint..<min(midpoint + 5, clampedCount) {
                sampleVals.append(String(format: "%.4f", audioSamples[i]))
            }
            let absMax = audioSamples.prefix(clampedCount).map { abs($0) }.max() ?? 0
            let mean = audioSamples.prefix(clampedCount).reduce(0.0, +) / Float(clampedCount)
            let statsText = String(
                format: "  Audio input: count=%d/%d, abs_max=%.4f, mean=%.6f",
                clampedCount, maxModelSamples, absMax, mean)
            logger.debug("\(statsText)")
            logger.debug("  mid_5=[\(sampleVals.joined(separator: ", "))]")
        }

        return (array, clampedCount)
    }

    private func makeAudioFeatureProvider(array: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: length)
        return try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: array),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])
    }

    private func makeEncoderInput(melFeatures: MLMultiArray, melLength: Int?) throws -> MLFeatureProvider {
        // The encoder expects:
        // - "melspectrogram_features": passthrough from MelSpectrogram
        // - "mel_length": [1] int32 frame count
        // Some exports also require a dummy "input_1": [1,1,1,1] fp16 flag.
        let lengthValue = melLength ?? melFeatures.shape.last?.intValue ?? 0
        guard lengthValue > 0 else {
            throw ASRError.processingFailed("Invalid mel_length for CTC encoder input")
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: lengthValue)

        var dict: [String: MLFeatureValue] = [
            "melspectrogram_features": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ]

        // Optional placeholder accepted by some staged exports.
        if let input1 = try? MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16) {
            input1[0] = 1
            dict["input_1"] = MLFeatureValue(multiArray: input1)
        }

        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    // MARK: - Overlap Merging

    /// Merge two overlapping CTC log-prob frames by taking the mean in
    /// probability space (numerically-stable logmeanexp).
    ///
    /// `logmeanexp(a, b) = max(a, b) + log(exp(a-m) + exp(b-m)) - log(2)`
    ///
    /// The previous implementation took the arithmetic mean of log-probs,
    /// which is `log(sqrt(p_a * p_b))` — a geometric mean that does not
    /// produce a valid probability distribution and that systematically
    /// underweights frames where the two chunks' predictions disagree.
    /// Both inputs `-inf` propagates to `-inf`.
    static func mergeOverlapFrame(existing: [Float], incoming: [Float]) -> [Float] {
        let v = min(existing.count, incoming.count)
        if v == 0 { return existing }

        let log2: Float = 0.69314718  // logf(2)
        var averaged = [Float](repeating: 0, count: v)
        for j in 0..<v {
            let a = existing[j]
            let b = incoming[j]
            let m = max(a, b)
            if m == -Float.infinity {
                averaged[j] = -Float.infinity
            } else {
                averaged[j] = m + logf(expf(a - m) + expf(b - m)) - log2
            }
        }
        return averaged
    }

    // MARK: - Log Probability Processing

    private func makeLogProbs(
        from ctcOutput: MLMultiArray,
        temperature: Float = 1.0,
        blankBias: Float = 0.0
    ) throws -> [[Float]] {
        let rank = ctcOutput.shape.count
        guard rank == 3 || rank == 4 else {
            throw ASRError.processingFailed("Unexpected CTC output rank: \(ctcOutput.shape)")
        }

        let shape = ctcOutput.shape.map { $0.intValue }
        let strides = ctcOutput.strides.map { $0.intValue }

        let vocabSize: Int
        let timeSteps: Int
        let timeStride: Int
        let vocabStride: Int

        if rank == 3 {
            // Expected shape: [1, timeSteps, vocabSize]
            timeSteps = shape[1]
            vocabSize = shape[2]
            timeStride = strides[1]
            vocabStride = strides[2]
        } else {
            // Expected shape: [1, vocabSize, 1, timeSteps]
            vocabSize = shape[1]
            timeSteps = shape[3]
            vocabStride = strides[1]
            timeStride = strides[3]
        }

        if vocabSize <= 0 || timeSteps <= 0 {
            return []
        }

        // Per-frame fp32 row buffer reused across iterations. We materialize
        // each frame contiguously (gathered by vocabStride) before passing it
        // to log-softmax / blank-bias / vDSP. This trades T allocations for
        // direct pointer access — orders of magnitude faster than the old
        // MLMultiArray subscript-by-NSNumber hot loop, which paid an ObjC
        // dispatch + array-allocation per element (T*V calls).
        var logProbs: [[Float]] = []
        logProbs.reserveCapacity(timeSteps)

        var rowBuffer = [Float](repeating: 0, count: vocabSize)

        switch ctcOutput.dataType {
        case .float32:
            let basePtr = ctcOutput.dataPointer.bindMemory(
                to: Float.self, capacity: ctcOutput.count)
            for t in 0..<timeSteps {
                rowBuffer.withUnsafeMutableBufferPointer { dst in
                    let dstPtr = dst.baseAddress!
                    if vocabStride == 1 {
                        // Contiguous along vocab — single memcpy.
                        dstPtr.update(from: basePtr.advanced(by: t * timeStride), count: vocabSize)
                    } else {
                        // Strided gather.
                        var src = basePtr.advanced(by: t * timeStride)
                        for v in 0..<vocabSize {
                            dstPtr[v] = src.pointee
                            src = src.advanced(by: vocabStride)
                        }
                    }
                }
                logProbs.append(
                    finalizeLogProbRow(
                        rowBuffer, temperature: temperature, blankBias: blankBias))
            }

        case .float16:
            // Read fp16 raw shorts and convert to fp32 per row using vImage.
            let basePtr = ctcOutput.dataPointer.bindMemory(
                to: UInt16.self, capacity: ctcOutput.count)
            // Temporary fp16 row when vocabStride != 1 (gather then convert).
            var fp16Row = [UInt16](repeating: 0, count: vocabSize)
            for t in 0..<timeSteps {
                let srcStart = basePtr.advanced(by: t * timeStride)
                fp16Row.withUnsafeMutableBufferPointer { fp16Buf in
                    let fp16Ptr = fp16Buf.baseAddress!
                    if vocabStride == 1 {
                        fp16Ptr.update(from: srcStart, count: vocabSize)
                    } else {
                        var src = srcStart
                        for v in 0..<vocabSize {
                            fp16Ptr[v] = src.pointee
                            src = src.advanced(by: vocabStride)
                        }
                    }
                    rowBuffer.withUnsafeMutableBufferPointer { fp32Buf in
                        var srcBuf = vImage_Buffer(
                            data: fp16Ptr,
                            height: 1,
                            width: vImagePixelCount(vocabSize),
                            rowBytes: vocabSize * MemoryLayout<UInt16>.stride
                        )
                        var dstBuf = vImage_Buffer(
                            data: fp32Buf.baseAddress!,
                            height: 1,
                            width: vImagePixelCount(vocabSize),
                            rowBytes: vocabSize * MemoryLayout<Float>.stride
                        )
                        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                    }
                }
                logProbs.append(
                    finalizeLogProbRow(
                        rowBuffer, temperature: temperature, blankBias: blankBias))
            }

        default:
            throw ASRError.processingFailed(
                "Unsupported CTC output dtype: \(ctcOutput.dataType.rawValue)")
        }

        return logProbs
    }

    /// Apply temperature, log-softmax, and blank bias to a single fp32 row.
    /// Returns a fresh `[Float]` so the caller can append it. Uses Accelerate
    /// for the softmax math.
    private func finalizeLogProbRow(
        _ row: [Float], temperature: Float, blankBias: Float
    ) -> [Float] {
        let n = row.count
        var scaled = row
        if temperature != 1.0 {
            var inv = 1.0 / temperature
            scaled.withUnsafeMutableBufferPointer { buf in
                vDSP_vsmul(buf.baseAddress!, 1, &inv, buf.baseAddress!, 1, vDSP_Length(n))
            }
        }

        // log-softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
        var maxVal: Float = 0
        scaled.withUnsafeBufferPointer { buf in
            vDSP_maxv(buf.baseAddress!, 1, &maxVal, vDSP_Length(n))
        }
        var negMax = -maxVal
        // shifted = scaled - max
        var shifted = [Float](repeating: 0, count: n)
        scaled.withUnsafeBufferPointer { src in
            shifted.withUnsafeMutableBufferPointer { dst in
                vDSP_vsadd(src.baseAddress!, 1, &negMax, dst.baseAddress!, 1, vDSP_Length(n))
            }
        }
        // exp(shifted)
        var expBuf = [Float](repeating: 0, count: n)
        shifted.withUnsafeBufferPointer { src in
            expBuf.withUnsafeMutableBufferPointer { dst in
                var count: Int32 = Int32(n)
                vvexpf(dst.baseAddress!, src.baseAddress!, &count)
            }
        }
        var sumExp: Float = 0
        expBuf.withUnsafeBufferPointer { buf in
            vDSP_sve(buf.baseAddress!, 1, &sumExp, vDSP_Length(n))
        }
        var negLogSumExp = -logf(sumExp)
        var result = [Float](repeating: 0, count: n)
        shifted.withUnsafeBufferPointer { src in
            result.withUnsafeMutableBufferPointer { dst in
                vDSP_vsadd(
                    src.baseAddress!, 1, &negLogSumExp, dst.baseAddress!, 1, vDSP_Length(n))
            }
        }

        if blankBias != 0.0 && blankId < n {
            result[blankId] -= blankBias
        }
        return result
    }

    private func logSoftmax(_ logits: [Float], temperature: Float = 1.0) -> [Float] {
        guard !logits.isEmpty else { return [] }

        // Apply temperature scaling: divide logits by temperature before softmax
        // Higher temperature = softer distribution (spreads probability mass)
        // Lower temperature = sharper distribution (more peaked)
        let scaledLogits = temperature != 1.0 ? logits.map { $0 / temperature } : logits

        let maxLogit = scaledLogits.max() ?? 0
        var sumExp: Float = 0

        for i in 0..<scaledLogits.count {
            sumExp += expf(scaledLogits[i] - maxLogit)
        }

        let logSumExp = logf(sumExp)
        var result: [Float] = Array(repeating: 0, count: scaledLogits.count)

        // log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
        for i in 0..<scaledLogits.count {
            result[i] = (scaledLogits[i] - maxLogit) - logSumExp
        }

        return result
    }

    private func trimLogProbs(_ logProbs: [[Float]], audioSampleCount: Int) -> [[Float]] {
        guard !logProbs.isEmpty else { return logProbs }

        let totalFrames = logProbs.count
        if audioSampleCount >= maxModelSamples {
            return logProbs
        }

        let samplesPerFrame = Double(maxModelSamples) / Double(totalFrames)
        let validFrames = Int(ceil(Double(audioSampleCount) / samplesPerFrame))
        let clampedFrames = max(1, min(validFrames, totalFrames))

        if debugMode {
            logger.debug("[DEBUG] Trimming CTC frames:")
            logger.debug(
                "[DEBUG]   totalFrames=\(totalFrames), sampleCount=\(audioSampleCount), maxModelSamples=\(maxModelSamples)"
            )
            logger.debug(
                "[DEBUG]   samplesPerFrame=\(String(format: "%.2f", samplesPerFrame)), validFrames=\(validFrames), clampedFrames=\(clampedFrames)"
            )
        }

        return Array(logProbs.prefix(clampedFrames))
    }
}
