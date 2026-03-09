import Foundation

/// Configuration and state for decode-time keyword biasing in the TDT decoder.
///
/// When provided to the decoder, token selection at each step is biased toward
/// keyword phrases defined in the prefix trie. Only tokens that appear in the
/// JointDecisionv2 top-k candidates can be boosted — no hallucination of
/// acoustically implausible tokens.
struct KeywordBoostingContext: Sendable {
    /// Prefix trie built from vocabulary terms with tokenIds.
    let prefixTrie: KeywordPrefixTrie
    /// Additive logit boost for keyword tokens (default from vocabulary alpha).
    let boostWeight: Float
    /// Optional real-time callback fired when a phrase is fully detected.
    let onPhraseDetected: (@Sendable (DetectedPhrase) -> Void)?

    /// Accumulated detected phrases during a decode pass.
    /// Populated by the decoder and surfaced in the ASRResult.
    var detectedPhrases: [DetectedPhrase] = []

    init(
        prefixTrie: KeywordPrefixTrie,
        boostWeight: Float = 3.0,
        onPhraseDetected: (@Sendable (DetectedPhrase) -> Void)? = nil
    ) {
        self.prefixTrie = prefixTrie
        self.boostWeight = boostWeight
        self.onPhraseDetected = onPhraseDetected
    }
}
