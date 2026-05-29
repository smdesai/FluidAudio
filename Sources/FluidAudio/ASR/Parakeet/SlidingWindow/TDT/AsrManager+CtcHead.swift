@preconcurrency import CoreML
import Foundation

extension AsrManager {

    public struct SharedCtcHeadResult: Sendable {
        public let logProbs: [[Float]]
        public let frameDuration: Double
        public let totalFrames: Int
        public let vocabulary: [Int: String]
    }

    /// Run the hybrid TDT-CTC shared encoder and standalone CTC head.
    ///
    /// This avoids loading or running the separate Parakeet CTC encoder when
    /// the active ASR model is `tdtCtc110m` and `CtcHead.mlmodelc` is available.
    public func computeSharedCtcHeadLogProbs(_ audioSamples: [Float]) async throws -> SharedCtcHeadResult? {
        guard let models = asrModels, models.version == .tdtCtc110m, let ctcHead = models.ctcHead else {
            return nil
        }
        guard !audioSamples.isEmpty else {
            return SharedCtcHeadResult(logProbs: [], frameDuration: 0, totalFrames: 0, vocabulary: models.vocabulary)
        }

        let actualLength = min(audioSamples.count, ASRConstants.maxModelSamples)
        let alignedSamples = Array(audioSamples.prefix(actualLength))
        let paddedAudio = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        let preprocessorInput = try await preparePreprocessorInput(paddedAudio, actualLength: actualLength)
        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        do {
            guard let preprocessorModel else {
                throw ASRError.notInitialized
            }
            let preprocessorOutput = try await preprocessorModel.compatPrediction(
                from: preprocessorInput,
                options: predictionOptions
            )
            let encoderOutput = try extractFeatureValue(
                from: preprocessorOutput,
                key: "encoder",
                errorMessage: "Invalid encoder output"
            )
            let encoderLength = try extractFeatureValue(
                from: preprocessorOutput,
                key: "encoder_length",
                errorMessage: "Invalid encoder output length"
            )[0].intValue

            let ctcInput = try makeCtcHeadInput(model: ctcHead, encoderOutput: encoderOutput)
            let ctcOutput = try await ctcHead.compatPrediction(from: ctcInput, options: predictionOptions)
            let ctcLogits = try extractCtcHeadLogits(from: ctcOutput)
            let blankId = models.vocabulary.count
            let logProbs = try CtcLogProbUtils.logProbs(
                from: ctcLogits,
                blankId: blankId,
                validFrames: encoderLength
            )
            let frameDuration =
                logProbs.isEmpty ? 0 : Double(actualLength) / Double(logProbs.count) / Double(config.sampleRate)

            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }

            return SharedCtcHeadResult(
                logProbs: logProbs,
                frameDuration: frameDuration,
                totalFrames: logProbs.count,
                vocabulary: models.vocabulary
            )
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    private nonisolated func makeCtcHeadInput(model: MLModel, encoderOutput: MLMultiArray) throws -> MLFeatureProvider {
        let inputNames = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard !inputNames.isEmpty else {
            throw ASRError.processingFailed("CtcHead model has no input descriptions")
        }
        let preferredName = inputNames.first { $0 == "encoder" } ?? inputNames[0]
        return try MLDictionaryFeatureProvider(dictionary: [
            preferredName: MLFeatureValue(multiArray: encoderOutput)
        ])
    }

    private nonisolated func extractCtcHeadLogits(from output: MLFeatureProvider) throws -> MLMultiArray {
        let names = ["ctc_logits", "ctc_head_raw_output", "ctc_head_output", "logits", "output"]
        for name in names {
            if let logits = output.featureValue(for: name)?.multiArrayValue {
                return logits
            }
        }
        if output.featureNames.count == 1, let name = output.featureNames.first,
            let logits = output.featureValue(for: name)?.multiArrayValue
        {
            return logits
        }
        throw ASRError.processingFailed(
            "CtcHead output missing logits. Available outputs: \(output.featureNames.sorted().joined(separator: ", "))"
        )
    }
}
