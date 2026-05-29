import Foundation

/// Configuration for ASR context biasing / custom vocabulary boosting.
public struct ContextBiasingConfig: Sendable {
    public enum CtcSource: Sendable {
        /// Prefer the shared TDT-CTC head when available, otherwise load the
        /// separate CTC model.
        case automatic
        /// Require the shared TDT-CTC head. If unavailable, no CTC fallback is used.
        case sharedHeadOnly
        /// Always use a separate CTC model.
        case separateCtc(CtcModelVariant)
    }

    public let vocabulary: CustomVocabularyContext
    public let ctcSource: CtcSource
    public let minSimilarity: Float?
    public let cbw: Float?
    public let marginSeconds: Double?
    public let rescorerConfig: VocabularyRescorer.Config

    public init(
        vocabulary: CustomVocabularyContext,
        ctcSource: CtcSource = .automatic,
        minSimilarity: Float? = nil,
        cbw: Float? = nil,
        marginSeconds: Double? = nil,
        rescorerConfig: VocabularyRescorer.Config = .default
    ) {
        self.vocabulary = vocabulary
        self.ctcSource = ctcSource
        self.minSimilarity = minSimilarity
        self.cbw = cbw
        self.marginSeconds = marginSeconds
        self.rescorerConfig = rescorerConfig
    }
}
