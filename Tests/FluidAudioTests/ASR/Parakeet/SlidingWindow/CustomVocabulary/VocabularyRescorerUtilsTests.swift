import XCTest

@testable import FluidAudio

final class VocabularyRescorerUtilsTests: XCTestCase {

    // MARK: - stringSimilarity

    func testIdenticalStrings() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("nvidia", "nvidia"), 1.0, accuracy: 0.01)
    }

    func testCompletelyDifferent() {
        // "abc" vs "xyz" -> distance 3, maxLen 3 -> sim = 0.0
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("abc", "xyz"), 0.0, accuracy: 0.01)
    }

    func testCaseInsensitive() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("NVIDIA", "nvidia"), 1.0, accuracy: 0.01)
    }

    func testOneCharDifference() {
        // "bose" vs "boz" -> distance 1 (e vs z) + length diff -> distance 2, maxLen 4
        // Actually: "bose" (4) vs "boz" (3) -> distance 2, maxLen 4 -> 1 - 2/4 = 0.5
        let sim = VocabularyRescorer.stringSimilarity("bose", "boz")
        XCTAssertEqual(sim, 0.5, accuracy: 0.01)
    }

    func testBothEmpty() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("", ""), 1.0, accuracy: 0.01)
    }

    func testOneEmpty() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("abc", ""), 0.0, accuracy: 0.01)
    }

    func testKnownPair() {
        // "nvida" vs "nvidia" -> distance 1, maxLen 6 -> 1 - 1/6 ≈ 0.833
        let sim = VocabularyRescorer.stringSimilarity("nvida", "nvidia")
        XCTAssertEqual(sim, 1.0 - 1.0 / 6.0, accuracy: 0.01)
    }

    // MARK: - lengthPenalizedSimilarity

    func testEqualLengthNoPenalty() {
        // Same length -> lengthRatio = 1.0 -> sqrt(1.0) = 1.0 -> no penalty
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("abcde", "abcde")
        let base = VocabularyRescorer.stringSimilarity("abcde", "abcde")
        XCTAssertEqual(lps, base, accuracy: 0.01)
    }

    func testShorterCompoundPenalized() {
        // "ab" (2) vs "abcdef" (6) -> lengthRatio = 2/6 ≈ 0.33
        // penalty = sqrt(0.33) ≈ 0.577
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("ab", "abcdef")
        let base = VocabularyRescorer.stringSimilarity("ab", "abcdef")
        XCTAssertLessThan(lps, base)
    }

    func testSameLengthSimilarWords() {
        // "newres" (6) vs "newrez" (6) -> equal length, sqrt(1.0) = 1.0
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("newres", "newrez")
        let base = VocabularyRescorer.stringSimilarity("newres", "newrez")
        XCTAssertEqual(lps, base, accuracy: 0.01)
    }

    // MARK: - normalizeForSimilarity

    func testNormalizeBasic() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Hello World!"), "hello world")
    }

    func testNormalizePreservesApostrophe() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("It's"), "it's")
    }

    func testNormalizePreservesHyphen() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Ramirez-Santos"), "ramirez-santos")
    }

    func testNormalizeMultipleSpaces() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("  hello   world  "), "hello world")
    }

    func testNormalizeEmptyString() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity(""), "")
    }

    func testNormalizeNumbers() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Test123"), "test123")
    }

    func testNormalizeTabsNewlines() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("hello\tworld\nfoo"), "hello world foo")
    }

    // MARK: - Config Adaptive Thresholds

    func testAdaptiveCbwAtReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 3), 3.0, accuracy: 0.01)
    }

    func testAdaptiveCbwLongerPhrase() {
        let config = VocabularyRescorer.Config.default
        // 6 tokens: ratio = 6/3 = 2.0, scaleFactor = 1.0 + log2(2.0)*0.3 = 1.3
        // result = 3.0 * 1.3 = 3.9
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 6), 3.9, accuracy: 0.01)
    }

    func testAdaptiveCbwBelowReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 2), 3.0, accuracy: 0.01)
    }

    func testAdaptiveCbwDisabled() {
        let config = VocabularyRescorer.Config(useAdaptiveThresholds: false)
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 10), 3.0, accuracy: 0.01)
    }

    // MARK: - Config Defaults

    func testConfigDefaultValues() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.useAdaptiveThresholds, ContextBiasingConstants.defaultUseAdaptiveThresholds)
        XCTAssertEqual(config.referenceTokenCount, ContextBiasingConstants.defaultReferenceTokenCount)
    }

    // MARK: - Stopword Sets

    func testMultiWordStopwordsExcludeContentWords() {
        // The multi-word path raises the threshold for spans containing
        // function words (the/and/of/etc.). It must NOT raise the
        // threshold on content words like "new"/"old"/"good"/"great" so
        // that rescues like `new red` → `Newrez` (sim 0.83) clear the
        // 0.55 floor.
        let contentWords = [
            "new", "old", "good", "great", "first", "last",
            "well", "back", "way", "own", "just", "also",
            "only", "even", "still", "now", "here",
            "there", "very",
        ]
        for word in contentWords {
            XCTAssertFalse(
                VocabularyRescorer.multiWordStopwords.contains(word),
                "'\(word)' should not be in multiWordStopwords (poisons multi-word rescue)"
            )
        }
    }

    func testMultiWordStopwordsIncludeFunctionWords() {
        // Function words still raise the threshold on multi-word spans.
        let functionWords = [
            "a", "the", "and", "or", "is", "to", "for",
            "in", "of", "with", "by", "i", "you", "he",
            "she", "it", "we", "they", "this", "that",
        ]
        for word in functionWords {
            XCTAssertTrue(
                VocabularyRescorer.multiWordStopwords.contains(word),
                "'\(word)' should be in multiWordStopwords"
            )
        }
    }

    func testSingleWordStopwordsRetainContentWords() {
        // Single-word path uses the wider list to avoid lone-word
        // substitutions like `just` → `Wyost`. Make sure the broader
        // set still includes those guards.
        let mustGuard = [
            "just", "new", "old", "good", "great", "back",
            "way", "own", "now", "here", "there", "still",
        ]
        for word in mustGuard {
            XCTAssertTrue(
                VocabularyRescorer.stopwords.contains(word),
                "'\(word)' should be in stopwords (single-word guard)"
            )
        }
    }

    func testStopwordAllowsHighSimilarityContentReplacement() async throws {
        let vocab = CustomVocabularyContext(terms: [])
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )

        let result = rescorer.checkStopwordRules(
            normalizedWord: "those",
            spanLength: 1,
            spanWords: [],
            vocabTerm: "Bose",
            currentSimilarity: 0.50
        )

        XCTAssertFalse(result.shouldSkip)
        XCTAssertGreaterThanOrEqual(result.adjustedMinSimilarity, 0.60)
    }

    func testStopwordRejectsLowSimilarityReplacement() async throws {
        let vocab = CustomVocabularyContext(terms: [])
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )

        let result = rescorer.checkStopwordRules(
            normalizedWord: "before",
            spanLength: 1,
            spanWords: [],
            vocabTerm: "Bose",
            currentSimilarity: 0.50
        )

        XCTAssertTrue(result.shouldSkip)
    }

    func testMultiWordSpanAnchoredEdgeAcceptsExactPhrase() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Felix Quinones")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertTrue(
            rescorer.multiWordSpanHasAnchoredEdge(
                spanWords: ["dr", "felix", "quinones"],
                forms: forms
            ))
    }

    func testMultiWordSpanAnchoredEdgeAcceptsCollapsedTrailingName() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Bao Halverson")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertTrue(
            rescorer.multiWordSpanHasAnchoredEdge(
                spanWords: ["dr", "bauhalversen"],
                forms: forms
            ))
    }

    func testAdjacentOmittedVocabEdgeRejectsDuplicatePrefix() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Aaron Petrov")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertTrue(
            rescorer.spanHasAdjacentOmittedVocabEdge(
                spanWords: ["aaron", "petrov"],
                previousWord: "dr",
                nextWord: nil,
                forms: forms
            ))
    }

    func testAdjacentOmittedVocabEdgeAllowsCollapsedNameWithoutPrefix() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Bao Halverson")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertFalse(
            rescorer.spanHasAdjacentOmittedVocabEdge(
                spanWords: ["dr", "bauhalversen"],
                previousWord: nil,
                nextWord: nil,
                forms: forms
            ))
    }

    func testMultiWordSpanAnchoredEdgeRejectsLeadingExtraWord() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Felix Quinones")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertFalse(
            rescorer.multiWordSpanHasAnchoredEdge(
                spanWords: ["to", "dr", "felix", "quinones"],
                forms: forms
            ))
    }

    func testMultiWordSpanAnchoredEdgeRejectsTrailingExtraWord() async throws {
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "Dr. Felix Quinones")]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )
        let forms = rescorer.buildNormalizedForms(for: vocab.terms[0])

        XCTAssertFalse(
            rescorer.multiWordSpanHasAnchoredEdge(
                spanWords: ["dr", "felix", "quinones", "reviewed"],
                forms: forms
            ))
    }

    func testPreserveCapitalizationKeepsCanonicalCasingAndTrailingPunctuation() async throws {
        let vocab = CustomVocabularyContext(terms: [])
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )

        XCTAssertEqual(
            rescorer.preserveCapitalization(original: "Hamachord,", replacement: "Hemacord"),
            "Hemacord,"
        )
        XCTAssertEqual(
            rescorer.preserveCapitalization(original: "Somovert.", replacement: "Somavert"),
            "Somavert."
        )
    }

    func testMultiWordComponentSetIncludesPhraseWords() async throws {
        let vocab = CustomVocabularyContext(
            terms: [
                CustomVocabularyTerm(text: "Dr. Aaron Petrov"),
                CustomVocabularyTerm(text: "Atryn"),
            ]
        )
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )

        let components = rescorer.buildMultiWordVocabularyComponentSet()

        XCTAssertTrue(components.contains("dr"))
        XCTAssertTrue(components.contains("aaron"))
        XCTAssertTrue(components.contains("petrov"))
        XCTAssertFalse(components.contains("atryn"))
    }

    func testShortWordThresholdTightensForLargeVocabulary() async throws {
        let terms =
            (0...ContextBiasingConstants.largeVocabThreshold).map { index in
                CustomVocabularyTerm(text: "Distractor\(index)")
            } + [CustomVocabularyTerm(text: "Atgam")]
        let vocab = CustomVocabularyContext(terms: terms)
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: vocab
        )

        let threshold = rescorer.checkLengthRatioRules(
            normalizedWord: "team",
            vocabTerm: "Atgam",
            currentSimilarity: 0.60,
            minSimilarity: 0.55
        )

        XCTAssertEqual(threshold, ContextBiasingConstants.largeVocabShortWordSimilarity, accuracy: 0.001)
    }

    // MARK: - buildWordTimings minConfidence aggregation

    func testWordTimingMinConfidenceIsMinOverTokens() async throws {
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: CustomVocabularyContext(terms: [])
        )
        // Two words: "▁me ps" (confident) then "▁ge v i" (one weak token 0.42).
        let tokens = [
            TokenTiming(token: "▁me", tokenId: 1, startTime: 0.0, endTime: 0.1, confidence: 0.99),
            TokenTiming(token: "ps", tokenId: 2, startTime: 0.1, endTime: 0.2, confidence: 0.95),
            TokenTiming(token: "▁ge", tokenId: 3, startTime: 0.3, endTime: 0.4, confidence: 0.88),
            TokenTiming(token: "v", tokenId: 4, startTime: 0.4, endTime: 0.5, confidence: 0.42),
            TokenTiming(token: "i", tokenId: 5, startTime: 0.5, endTime: 0.6, confidence: 0.91),
        ]
        let words = rescorer.buildWordTimings(from: tokens)
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "meps")
        XCTAssertEqual(words[0].minConfidence, 0.95, accuracy: 0.001, "min of 0.99, 0.95")
        XCTAssertEqual(words[1].word, "gevi")
        XCTAssertEqual(words[1].minConfidence, 0.42, accuracy: 0.001, "min of 0.88, 0.42, 0.91")
    }

    func testWordTimingMinConfidenceDefaultsHighForSingleConfidentToken() async throws {
        let rescorer = try await VocabularyRescorer.create(
            spotter: CtcKeywordSpotter(vocabulary: [:]),
            vocabulary: CustomVocabularyContext(terms: [])
        )
        let tokens = [
            TokenTiming(token: "▁prior", tokenId: 1, startTime: 0.0, endTime: 0.2, confidence: 1.0)
        ]
        let words = rescorer.buildWordTimings(from: tokens)
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].minConfidence, 1.0, accuracy: 0.001)
    }
}
