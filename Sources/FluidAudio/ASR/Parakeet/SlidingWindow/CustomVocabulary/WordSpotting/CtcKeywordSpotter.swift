import CoreML
import Foundation

/// Swift implementation of CTC keyword spotting for Parakeet-TDT CTC 110M,
/// mirroring the NeMo `ctc_word_spot` dynamic programming algorithm.
///
/// This engine:
/// - Runs the MelSpectrogram + AudioEncoder CoreML models from `CtcModels`.
/// - Extracts CTC logits and converts them to log-probabilities over time.
/// - Applies DP to score each keyword independently (no beam search competition).
public struct CtcKeywordSpotter: Sendable {

    let logger = AppLogger(category: "CtcKeywordSpotter")
    let models: CtcModels?
    let vocabulary: [Int: String]
    public let blankId: Int

    /// Computed property to avoid storing non-Sendable MLPredictionOptions.
    /// Creating on demand is cheap (just init + empty dict).
    var predictionOptions: MLPredictionOptions {
        AsrModels.optimizedPredictionOptions()
    }

    let sampleRate: Int = ASRConstants.sampleRate
    let maxModelSamples: Int = ASRConstants.maxModelSamples

    // Chunking parameters for audio longer than maxModelSamples
    // 2s overlap at 16kHz = 32,000 samples (matches TDT chunking pattern)
    let chunkOverlapSamples: Int = 32_000

    // Debug flag - enabled only in DEBUG builds
    #if DEBUG
    let debugMode: Bool = true  // Set to true locally for verbose logging
    #else
    let debugMode: Bool = false
    #endif

    // Temperature for CTC softmax (higher = softer distribution, lower = more peaked)
    let temperature: Float = ContextBiasingConstants.ctcTemperature

    // Blank bias applied to log probabilities (positive values penalize blank token)
    let blankBias: Float = ContextBiasingConstants.blankBias

    // MARK: - Result Types

    struct CtcLogProbResult: Sendable {
        let logProbs: [[Float]]
        let frameDuration: Double
        let totalFrames: Int
        let audioSamplesUsed: Int
    }

    /// Public result type containing detections and cached CTC log-probabilities.
    /// The log-probs can be reused for scoring additional words without re-running the CTC model.
    public struct SpotKeywordsResult: Sendable {
        public init(
            detections: [KeywordDetection],
            wordAlignments: [CtcWordAlignment],
            logProbs: [[Float]],
            frameDuration: Double,
            totalFrames: Int
        ) {
            self.detections = detections
            self.wordAlignments = wordAlignments
            self.logProbs = logProbs
            self.frameDuration = frameDuration
            self.totalFrames = totalFrames
        }

        /// Keyword detections for vocabulary terms
        public let detections: [KeywordDetection]
        /// Greedy CTC word alignment used as the baseline for context-biasing candidates.
        public let wordAlignments: [CtcWordAlignment]
        /// CTC log-probabilities [T, V] for reuse in rescoring
        public let logProbs: [[Float]]
        /// Duration of each CTC frame in seconds
        public let frameDuration: Double
        /// Total number of CTC frames
        public let totalFrames: Int
    }

    /// Result for a single keyword detection.
    public struct KeywordDetection: Sendable {
        public let term: CustomVocabularyTerm
        public let score: Float
        public let totalFrames: Int
        public let startFrame: Int
        public let endFrame: Int
        public let startTime: TimeInterval
        public let endTime: TimeInterval

        public init(
            term: CustomVocabularyTerm,
            score: Float,
            totalFrames: Int,
            startFrame: Int,
            endFrame: Int,
            startTime: TimeInterval,
            endTime: TimeInterval
        ) {
            self.term = term
            self.score = score
            self.totalFrames = totalFrames
            self.startFrame = startFrame
            self.endFrame = endFrame
            self.startTime = startTime
            self.endTime = endTime
        }
    }

    public init(models: CtcModels, blankId: Int = ContextBiasingConstants.defaultBlankId) {
        self.models = models
        self.vocabulary = models.vocabulary
        self.blankId = blankId
    }

    /// Initialize a spotter for precomputed CTC log-probabilities.
    ///
    /// This path is used by the shared TDT-CTC head integration, where the
    /// ASR encoder has already produced CTC logits and no separate CTC model
    /// bundle should be loaded just to run graph search and constrained DP.
    public init(vocabulary: [Int: String], blankId: Int = ContextBiasingConstants.defaultBlankId) {
        self.models = nil
        self.vocabulary = vocabulary
        self.blankId = blankId
    }

    // MARK: - Public API

    /// Spot keywords and return both detections and cached log-probabilities.
    /// The log-probs can be reused for scoring additional words (e.g., original transcript words)
    /// without re-running the expensive CTC model inference.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono audio samples.
    ///   - customVocabulary: Vocabulary context with pre-tokenized terms.
    ///   - minScore: Optional minimum score threshold for detections.
    /// - Returns: SpotKeywordsResult containing detections and reusable log-probs.
    public func spotKeywordsWithLogProbs(
        audioSamples: [Float],
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) async throws -> SpotKeywordsResult {
        let ctcProfileEnabled =
            ProcessInfo.processInfo.environment["FA_CTC_PROFILE"] != nil

        let inferT0 = Date()
        let ctcResult = try await computeLogProbs(for: audioSamples)
        let inferDt = Date().timeIntervalSince(inferT0)
        let logProbs = ctcResult.logProbs
        guard !logProbs.isEmpty else {
            return SpotKeywordsResult(
                detections: [], wordAlignments: [], logProbs: [], frameDuration: 0, totalFrames: 0)
        }

        let frameDuration = ctcResult.frameDuration
        let totalFrames = ctcResult.totalFrames

        let dpT0 = Date()
        let results = spotKeywordsFromLogProbs(
            logProbs: logProbs,
            frameDuration: frameDuration,
            customVocabulary: customVocabulary,
            minScore: minScore
        ).detections

        if ctcProfileEnabled {
            let dpDt = Date().timeIntervalSince(dpT0)
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "CTC-PROFILE inference=%.3fs (%d frames) DP=%.3fs (%d terms scored, %d detections)\n",
                        inferDt, totalFrames, dpDt, customVocabulary.terms.count, results.count
                    ).utf8))
        }

        return SpotKeywordsResult(
            detections: results,
            wordAlignments: CtcWordAligner.align(
                logProbs: logProbs,
                vocabulary: vocabulary,
                blankId: blankId,
                frameDuration: frameDuration
            ),
            logProbs: logProbs,
            frameDuration: frameDuration,
            totalFrames: totalFrames
        )
    }

    /// Spot keywords using pre-computed log-probabilities (no CTC inference).
    /// Use this when CTC logProbs are already available (e.g. from a unified Preprocessor
    /// that exports CTC logits alongside encoder features).
    ///
    /// - Parameters:
    ///   - logProbs: Pre-computed CTC log-probabilities [T, V].
    ///   - frameDuration: Duration of each CTC frame in seconds.
    ///   - customVocabulary: Vocabulary context with pre-tokenized terms.
    ///   - minScore: Optional minimum score threshold for detections.
    /// - Returns: SpotKeywordsResult containing detections and the same log-probs passed in.
    public func spotKeywordsFromLogProbs(
        logProbs: [[Float]],
        frameDuration: Double,
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) -> SpotKeywordsResult {
        let totalFrames = logProbs.count
        guard totalFrames > 0 else {
            return SpotKeywordsResult(
                detections: [], wordAlignments: [], logProbs: [], frameDuration: 0, totalFrames: 0)
        }

        let graphVocab = CustomVocabularyContext(
            terms: customVocabulary.terms.filter { term in
                guard let tokenIds = term.ctcTokenIds ?? term.tokenIds else { return false }
                return !tokenIds.contains(ContextBiasingConstants.wildcardTokenId)
            },
            alpha: customVocabulary.alpha,
            minCtcScore: customVocabulary.minCtcScore,
            minSimilarity: customVocabulary.minSimilarity,
            minCombinedConfidence: customVocabulary.minCombinedConfidence,
            minTermLength: customVocabulary.minTermLength
        )
        let graph = CtcContextGraph(
            vocabulary: graphVocab,
            minScore: minScore,
            blankId: blankId,
            contextScore: 0
        )
        let graphDetections = graph.spot(logProbs: logProbs)
        var results = graphDetections.map { detection in
            KeywordDetection(
                term: detection.entry.term,
                score: detection.score,
                totalFrames: totalFrames,
                startFrame: detection.startFrame,
                endFrame: detection.endFrame,
                startTime: TimeInterval(detection.startFrame) * frameDuration,
                endTime: TimeInterval(detection.endFrame) * frameDuration
            )
        }

        for term in customVocabulary.terms {
            guard let ids = term.ctcTokenIds ?? term.tokenIds else { continue }
            guard ids.contains(ContextBiasingConstants.wildcardTokenId) else { continue }
            guard term.text.count >= customVocabulary.minTermLength, !ids.isEmpty else { continue }

            let adjustedThreshold = Self.adjustedSpotterThreshold(minScore, tokenCount: ids.count)
            let multipleDetections = ctcWordSpotMultiple(
                logProbs: logProbs,
                keywordTokens: ids,
                minScore: adjustedThreshold,
                mergeOverlap: true
            )
            for (score, start, end) in multipleDetections {
                results.append(
                    KeywordDetection(
                        term: term,
                        score: score,
                        totalFrames: totalFrames,
                        startFrame: start,
                        endFrame: end,
                        startTime: TimeInterval(start) * frameDuration,
                        endTime: TimeInterval(end) * frameDuration
                    ))
            }
        }

        return SpotKeywordsResult(
            detections: results,
            wordAlignments: CtcWordAligner.align(
                logProbs: logProbs,
                vocabulary: vocabulary,
                blankId: blankId,
                frameDuration: frameDuration
            ),
            logProbs: logProbs,
            frameDuration: frameDuration,
            totalFrames: totalFrames
        )
    }

    // MARK: - Log-Probability Conversion

    /// Convert raw CTC logits to log-probabilities with temperature scaling and blank bias.
    /// Use this to post-process raw logits from a unified Preprocessor before passing to
    /// `spotKeywordsFromLogProbs` or `VocabularyRescorer.ctcTokenRescore`.
    ///
    /// - Parameters:
    ///   - rawLogits: Raw CTC logits [T, V] (before softmax).
    ///   - blankId: Index of the blank token in the vocabulary.
    ///   - temperature: Temperature for softmax scaling (default from ContextBiasingConstants).
    ///   - blankBias: Penalty applied to blank token log-probability (default from ContextBiasingConstants).
    /// - Returns: Log-probabilities [T, V] after log-softmax, temperature, and blank bias.
    public static func applyLogSoftmax(
        rawLogits: [[Float]],
        blankId: Int,
        temperature: Float = ContextBiasingConstants.ctcTemperature,
        blankBias: Float = ContextBiasingConstants.blankBias
    ) -> [[Float]] {
        var logProbs = [[Float]]()
        logProbs.reserveCapacity(rawLogits.count)

        for logits in rawLogits {
            guard !logits.isEmpty else {
                logProbs.append([])
                continue
            }

            // Temperature scaling
            let scaled = temperature != 1.0 ? logits.map { $0 / temperature } : logits

            // Log-softmax
            let maxVal = scaled.max() ?? 0
            var sumExp: Float = 0
            for v in scaled { sumExp += expf(v - maxVal) }
            let logSumExp = logf(sumExp)

            var row = [Float](repeating: 0, count: scaled.count)
            for i in 0..<scaled.count {
                row[i] = (scaled[i] - maxVal) - logSumExp
            }

            // Blank bias
            if blankBias != 0.0 && blankId < row.count {
                row[blankId] -= blankBias
            }

            logProbs.append(row)
        }

        return logProbs
    }

    // MARK: - NeMo-compatible DP (delegated to CtcDPAlgorithm)

    func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            searchStartFrame: searchStartFrame,
            searchEndFrame: searchEndFrame,
            blankId: blankId
        )
    }

    func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        CtcDPAlgorithm.ctcWordSpotMultiple(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            minScore: minScore,
            mergeOverlap: mergeOverlap,
            blankId: blankId
        )
    }

    private static func adjustedSpotterThreshold(_ minScore: Float?, tokenCount: Int) -> Float {
        let base = minScore ?? ContextBiasingConstants.defaultMinSpotterScore
        let extraTokens = max(0, tokenCount - ContextBiasingConstants.baselineTokenCountForThreshold)
        return base - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
    }

}
