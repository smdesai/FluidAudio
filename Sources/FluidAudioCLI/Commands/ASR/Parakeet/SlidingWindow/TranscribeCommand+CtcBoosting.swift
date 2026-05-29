import Foundation

import FluidAudio

struct VocabularyBoostingInputs {
    let vocabulary: CustomVocabularyContext
    let spotter: CtcKeywordSpotter
    let spotResult: CtcKeywordSpotter.SpotKeywordsResult
    let ctcModelDirectory: URL
    let usedSharedCtcHead: Bool
    let loadDuration: TimeInterval
    let spotDuration: TimeInterval
}

func prepareVocabularyBoostingInputs(
    vocabPath: String,
    audioSamples: [Float],
    asrManager: AsrManager,
    logger: AppLogger
) async throws -> VocabularyBoostingInputs {
    let loadT0 = Date()
    if let shared = try await asrManager.computeSharedCtcHeadLogProbs(audioSamples) {
        let vocab = try await CustomVocabularyContext.loadWithCtcTokensOnly(from: vocabPath)
        let spotter = CtcKeywordSpotter(vocabulary: shared.vocabulary, blankId: shared.vocabulary.count)
        let loadDuration = Date().timeIntervalSince(loadT0)
        let spotT0 = Date()
        let spotResult = spotter.spotKeywordsFromLogProbs(
            logProbs: shared.logProbs,
            frameDuration: shared.frameDuration,
            customVocabulary: vocab,
            minScore: nil
        )
        let spotDuration = Date().timeIntervalSince(spotT0)
        logger.info("Using shared TDT-CTC CTC head for vocabulary boosting")
        return VocabularyBoostingInputs(
            vocabulary: vocab,
            spotter: spotter,
            spotResult: spotResult,
            ctcModelDirectory: CtcModels.defaultCacheDirectory(for: .ctc110m),
            usedSharedCtcHead: true,
            loadDuration: loadDuration,
            spotDuration: spotDuration
        )
    }

    let (vocab, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(from: vocabPath)
    let loadDuration = Date().timeIntervalSince(loadT0)
    let spotter = CtcKeywordSpotter(models: ctcModels, blankId: ctcModels.vocabulary.count)
    let spotT0 = Date()
    let spotResult = try await spotter.spotKeywordsWithLogProbs(
        audioSamples: audioSamples,
        customVocabulary: vocab,
        minScore: nil
    )
    let spotDuration = Date().timeIntervalSince(spotT0)

    return VocabularyBoostingInputs(
        vocabulary: vocab,
        spotter: spotter,
        spotResult: spotResult,
        ctcModelDirectory: CtcModels.defaultCacheDirectory(for: ctcModels.variant),
        usedSharedCtcHead: false,
        loadDuration: loadDuration,
        spotDuration: spotDuration
    )
}
