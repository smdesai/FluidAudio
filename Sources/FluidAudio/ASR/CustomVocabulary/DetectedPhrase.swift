import Foundation

/// A keyword phrase detected during TDT decode-time biasing.
public struct DetectedPhrase: Sendable, Codable {
    /// The matched vocabulary term.
    public let term: CustomVocabularyTerm
    /// Start time of the phrase in the audio (seconds).
    public let startTime: TimeInterval
    /// End time of the phrase in the audio (seconds).
    public let endTime: TimeInterval
    /// Confidence indicator: lower values mean the greedy path already matched (more confident).
    /// Higher values mean more boosting was required.
    public let confidence: Float
    /// True if the token selection was overridden by boosting; false if greedy already matched.
    public let wasBoosted: Bool

    public init(
        term: CustomVocabularyTerm,
        startTime: TimeInterval,
        endTime: TimeInterval,
        confidence: Float,
        wasBoosted: Bool
    ) {
        self.term = term
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
        self.wasBoosted = wasBoosted
    }
}
