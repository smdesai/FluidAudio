import AVFoundation
import Foundation

extension AsrManager {

    public func transcribe(
        _ audioBuffer: AVAudioPCMBuffer,
        decoderState: inout TdtDecoderState,
        language: Language? = nil,
        contextBiasing: ContextBiasingConfig
    ) async throws -> ASRResult {
        let audioFloatArray = try AudioConverter().resampleBuffer(audioBuffer)
        return try await transcribe(
            audioFloatArray,
            decoderState: &decoderState,
            language: language,
            contextBiasing: contextBiasing
        )
    }

    public func transcribe(
        _ url: URL,
        decoderState: inout TdtDecoderState,
        language: Language? = nil,
        contextBiasing: ContextBiasingConfig
    ) async throws -> ASRResult {
        let audioFloatArray = try AudioConverter().resampleAudioFile(url)
        return try await transcribe(
            audioFloatArray,
            decoderState: &decoderState,
            language: language,
            contextBiasing: contextBiasing
        )
    }

    public func transcribe(
        _ audioSamples: [Float],
        decoderState: inout TdtDecoderState,
        language: Language? = nil,
        contextBiasing: ContextBiasingConfig
    ) async throws -> ASRResult {
        let result = try await transcribe(audioSamples, decoderState: &decoderState, language: language)
        return try await applyContextBiasing(
            to: result,
            audioSamples: audioSamples,
            contextBiasing: contextBiasing
        )
    }

    private func applyContextBiasing(
        to result: ASRResult,
        audioSamples: [Float],
        contextBiasing: ContextBiasingConfig
    ) async throws -> ASRResult {
        guard let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty else { return result }

        guard
            let inputs = try await makeContextBiasingInputs(
                audioSamples: audioSamples,
                config: contextBiasing
            )
        else {
            return result
        }
        guard !inputs.spotResult.logProbs.isEmpty else { return result }

        let vocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: inputs.vocabulary.terms.count)
        let minSimilarity =
            contextBiasing.minSimilarity ?? max(vocabConfig.minSimilarity, inputs.vocabulary.minSimilarity)
        let cbw = contextBiasing.cbw ?? vocabConfig.cbw
        let marginSeconds = contextBiasing.marginSeconds ?? ContextBiasingConstants.defaultMarginSeconds

        let rescorer = try await VocabularyRescorer.create(
            spotter: inputs.spotter,
            vocabulary: inputs.vocabulary,
            config: contextBiasing.rescorerConfig,
            ctcModelDirectory: inputs.ctcModelDirectory
        )
        let rescoreOutput = rescorer.ctcTokenRescore(
            transcript: result.text,
            tokenTimings: tokenTimings,
            logProbs: inputs.spotResult.logProbs,
            frameDuration: inputs.spotResult.frameDuration,
            ctcWordAlignments: inputs.spotResult.wordAlignments,
            cbw: cbw,
            marginSeconds: marginSeconds,
            minSimilarity: minSimilarity
        )
        guard rescoreOutput.wasModified else { return result }

        let detected = rescoreOutput.replacements.compactMap { $0.replacementWord }
        let applied = rescoreOutput.replacements.filter { $0.shouldReplace }.compactMap { $0.replacementWord }
        return result.withRescoring(
            text: rescoreOutput.text,
            detected: detected.isEmpty ? nil : detected,
            applied: applied.isEmpty ? nil : applied
        )
    }

    private struct ContextBiasingInputs {
        let vocabulary: CustomVocabularyContext
        let spotter: CtcKeywordSpotter
        let spotResult: CtcKeywordSpotter.SpotKeywordsResult
        let ctcModelDirectory: URL
    }

    private func makeContextBiasingInputs(
        audioSamples: [Float],
        config: ContextBiasingConfig
    ) async throws -> ContextBiasingInputs? {
        switch config.ctcSource {
        case .automatic:
            if let shared = try await makeSharedHeadInputs(audioSamples: audioSamples, vocabulary: config.vocabulary) {
                return shared
            }
            return try await makeSeparateCtcInputs(
                audioSamples: audioSamples, vocabulary: config.vocabulary, variant: .ctc110m)

        case .sharedHeadOnly:
            return try await makeSharedHeadInputs(audioSamples: audioSamples, vocabulary: config.vocabulary)

        case .separateCtc(let variant):
            return try await makeSeparateCtcInputs(
                audioSamples: audioSamples, vocabulary: config.vocabulary, variant: variant)
        }
    }

    private func makeSharedHeadInputs(
        audioSamples: [Float],
        vocabulary: CustomVocabularyContext
    ) async throws -> ContextBiasingInputs? {
        guard let shared = try await computeSharedCtcHeadLogProbs(audioSamples) else { return nil }
        let spotter = CtcKeywordSpotter(vocabulary: shared.vocabulary, blankId: shared.vocabulary.count)
        let spotResult = spotter.spotKeywordsFromLogProbs(
            logProbs: shared.logProbs,
            frameDuration: shared.frameDuration,
            customVocabulary: vocabulary,
            minScore: nil
        )
        return ContextBiasingInputs(
            vocabulary: vocabulary,
            spotter: spotter,
            spotResult: spotResult,
            ctcModelDirectory: CtcModels.defaultCacheDirectory(for: .ctc110m)
        )
    }

    private func makeSeparateCtcInputs(
        audioSamples: [Float],
        vocabulary: CustomVocabularyContext,
        variant: CtcModelVariant
    ) async throws -> ContextBiasingInputs {
        let ctcModels = try await CtcModels.downloadAndLoad(variant: variant)
        let spotter = CtcKeywordSpotter(models: ctcModels, blankId: ctcModels.vocabulary.count)
        let spotResult = try await spotter.spotKeywordsWithLogProbs(
            audioSamples: audioSamples,
            customVocabulary: vocabulary,
            minScore: nil
        )
        return ContextBiasingInputs(
            vocabulary: vocabulary,
            spotter: spotter,
            spotResult: spotResult,
            ctcModelDirectory: CtcModels.defaultCacheDirectory(for: variant)
        )
    }
}
