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

func shouldRunAutomaticTdtVeto(
    customVocab: CustomVocabularyContext,
    asrModels: AsrModels?,
    disableAutoTdtVeto: Bool,
    tdtPrimary: Bool,
    beamSize: Int
) -> Bool {
    guard !disableAutoTdtVeto else { return false }
    guard !tdtPrimary, beamSize == 0 else { return false }
    guard customVocab.terms.count > ContextBiasingConstants.largeVocabThreshold else { return false }
    guard let asrModels, asrModels.jointLogits != nil else { return false }
    return true
}

func prepareBeamBiasConfig(
    vocabPath: String?,
    audioSamples: [Float],
    asrManager: AsrManager,
    asrModels: AsrModels,
    bonus: Float,
    logger: AppLogger
) async throws -> (vocabulary: CustomVocabularyContext?, bias: TdtBeamBiasConfig?) {
    guard let vocabPath else { return (nil, nil) }

    let boostingInputs = try await prepareVocabularyBoostingInputs(
        vocabPath: vocabPath,
        audioSamples: audioSamples,
        asrManager: asrManager,
        logger: logger
    )
    let vocab = boostingInputs.vocabulary

    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")
    let tokenizer = try SentencePieceTokenizer(modelData: try Data(contentsOf: tokenizerURL))

    var keywordTokenSequences: [[Int]] = []
    var keptTermIndices: [Int] = []
    for (index, term) in vocab.terms.enumerated() {
        let ids = tokenizer.encode(term.text)
        if !ids.isEmpty {
            keywordTokenSequences.append(ids)
            keptTermIndices.append(index)
        }
    }
    guard !keywordTokenSequences.isEmpty else { return (vocab, nil) }

    let beamIndexForTermIndex = Dictionary(uniqueKeysWithValues: keptTermIndices.enumerated().map { ($1, $0) })
    let detectionSlopFrames = 4
    let windows = boostingInputs.spotResult.detections.compactMap { detection -> TdtBeamBiasWindow? in
        guard let termIndex = vocab.terms.firstIndex(where: { $0.text == detection.term.text }) else { return nil }
        guard let beamIndex = beamIndexForTermIndex[termIndex] else { return nil }
        return TdtBeamBiasWindow(
            keywordIndex: beamIndex,
            startFrame: max(0, detection.startFrame - detectionSlopFrames),
            endFrame: detection.endFrame + detectionSlopFrames
        )
    }

    logger.info(
        "Beam: loaded \(vocab.terms.count) vocab terms, \(keywordTokenSequences.count) tokenized, \(windows.count) CTC windows"
    )
    return (
        vocab,
        TdtBeamBiasConfig(
            keywordTokenSequences: keywordTokenSequences,
            bonus: bonus,
            windows: windows
        )
    )
}
