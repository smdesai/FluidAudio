import XCTest

@testable import FluidAudio

final class KeywordBoostingTests: XCTestCase {

    // MARK: - KeywordPrefixTrie Tests

    func testTrieBuildFromTermsWithTokenIds() {
        let terms = [
            CustomVocabularyTerm(text: "cancel", tokenIds: [10, 20, 30]),
            CustomVocabularyTerm(text: "order", tokenIds: [40, 50]),
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        XCTAssertFalse(trie.isEmpty)
    }

    func testTrieSkipsTermsWithoutTokenIds() {
        let terms = [
            CustomVocabularyTerm(text: "no tokens"),
            CustomVocabularyTerm(text: "has tokens", tokenIds: [1, 2]),
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        XCTAssertFalse(trie.isEmpty)

        // Only the term with tokenIds should be in the trie
        let validFromRoot = trie.getValidNextTokens(prefix: [])
        XCTAssertTrue(validFromRoot.contains(1))
        XCTAssertFalse(validFromRoot.contains(0))
    }

    func testTrieEmptyWhenNoTokenIds() {
        let terms = [
            CustomVocabularyTerm(text: "no tokens")
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        XCTAssertTrue(trie.isEmpty)
    }

    func testTrieCompletePhraseDetection() {
        let terms = [
            CustomVocabularyTerm(text: "hello", tokenIds: [10, 20])
        ]
        let trie = KeywordPrefixTrie(terms: terms)

        // Partial match — not complete
        let partial = trie.isComplete(prefix: [10])
        XCTAssertFalse(partial.matched)
        XCTAssertNil(partial.termIndex)

        // Full match
        let full = trie.isComplete(prefix: [10, 20])
        XCTAssertTrue(full.matched)
        XCTAssertEqual(full.termIndex, 0)
    }

    func testTrieNoMatchForWrongTokens() {
        let terms = [
            CustomVocabularyTerm(text: "hello", tokenIds: [10, 20])
        ]
        let trie = KeywordPrefixTrie(terms: terms)

        let result = trie.isComplete(prefix: [99, 88])
        XCTAssertFalse(result.matched)
    }

    func testTrieMultipleTerms() {
        let terms = [
            CustomVocabularyTerm(text: "cancel", tokenIds: [10, 20]),
            CustomVocabularyTerm(text: "confirm", tokenIds: [10, 30]),
        ]
        let trie = KeywordPrefixTrie(terms: terms)

        // Both share prefix token 10
        let validFromRoot = trie.getValidNextTokens(prefix: [])
        XCTAssertEqual(validFromRoot, [10])

        // After token 10, both 20 and 30 are valid
        let validAfter10 = trie.getValidNextTokens(prefix: [10])
        XCTAssertTrue(validAfter10.contains(20))
        XCTAssertTrue(validAfter10.contains(30))

        // Complete first term
        let cancel = trie.isComplete(prefix: [10, 20])
        XCTAssertTrue(cancel.matched)
        XCTAssertEqual(cancel.termIndex, 0)

        // Complete second term
        let confirm = trie.isComplete(prefix: [10, 30])
        XCTAssertTrue(confirm.matched)
        XCTAssertEqual(confirm.termIndex, 1)
    }

    // MARK: - TrieCursor Tests

    func testCursorAdvance() {
        let terms = [
            CustomVocabularyTerm(text: "abc", tokenIds: [1, 2, 3])
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        let cursor = trie.makeCursor()

        // Advance with valid token
        let next = cursor.advance(token: 1)
        XCTAssertNotNil(next)
        XCTAssertEqual(next?.prefix, [1])
        XCTAssertFalse(next?.isTerminal ?? true)

        // Advance further
        let next2 = next?.advance(token: 2)
        XCTAssertNotNil(next2)
        XCTAssertEqual(next2?.prefix, [1, 2])
        XCTAssertFalse(next2?.isTerminal ?? true)

        // Complete the phrase
        let final = next2?.advance(token: 3)
        XCTAssertNotNil(final)
        XCTAssertTrue(final?.isTerminal ?? false)
        XCTAssertEqual(final?.matchedTermIndex, 0)
    }

    func testCursorAdvanceInvalidToken() {
        let terms = [
            CustomVocabularyTerm(text: "abc", tokenIds: [1, 2, 3])
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        let cursor = trie.makeCursor()

        let invalid = cursor.advance(token: 99)
        XCTAssertNil(invalid)
    }

    func testCursorValidNextTokens() {
        let terms = [
            CustomVocabularyTerm(text: "a", tokenIds: [1, 2]),
            CustomVocabularyTerm(text: "b", tokenIds: [1, 3]),
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        let cursor = trie.makeCursor()

        XCTAssertEqual(cursor.validNextTokens, [1])

        let after1 = cursor.advance(token: 1)!
        XCTAssertEqual(after1.validNextTokens, [2, 3])
    }

    // MARK: - TdtGreedyTokenizer Tests

    func testTokenizerGreedyEncode() {
        // Build a simple vocabulary
        let vocab: [Int: String] = [
            0: " the",
            1: " cat",
            2: " c",
            3: "at",
        ]
        let tokenizer = TdtGreedyTokenizer(vocabulary: vocab)

        // Greedy longest-match: " the" → [0]
        let result = tokenizer.encode("the")
        XCTAssertEqual(result, [0])
    }

    func testTokenizerAllPaths() {
        let vocab: [Int: String] = [
            0: " ca",
            1: "t",
            2: " cat",
        ]
        let tokenizer = TdtGreedyTokenizer(vocabulary: vocab)

        let paths = tokenizer.encodeAllPaths("cat")
        XCTAssertFalse(paths.isEmpty)

        // Should include both decompositions: [2] and [0, 1]
        XCTAssertTrue(paths.contains([2]))
        XCTAssertTrue(paths.contains([0, 1]))
    }

    func testTokenizerEmptyInput() {
        let vocab: [Int: String] = [0: " a"]
        let tokenizer = TdtGreedyTokenizer(vocabulary: vocab)
        XCTAssertTrue(tokenizer.encode("").isEmpty)
        XCTAssertTrue(tokenizer.encodeAllPaths("").isEmpty)
    }

    func testTokenizerSkipsSpecialTokens() {
        let vocab: [Int: String] = [
            0: "<blank>",
            1: " hi",
        ]
        let tokenizer = TdtGreedyTokenizer(vocabulary: vocab)
        let result = tokenizer.encode("hi")
        XCTAssertEqual(result, [1])
    }

    // MARK: - KeywordBoostingContext Tests

    func testBoostingContextCreation() {
        let terms = [
            CustomVocabularyTerm(text: "test", tokenIds: [1, 2])
        ]
        let trie = KeywordPrefixTrie(terms: terms)
        let context = KeywordBoostingContext(prefixTrie: trie, boostWeight: 5.0)

        XCTAssertEqual(context.boostWeight, 5.0)
        XCTAssertTrue(context.detectedPhrases.isEmpty)
    }

    // MARK: - DetectedPhrase Tests

    func testDetectedPhraseCreation() {
        let term = CustomVocabularyTerm(text: "NVIDIA", tokenIds: [10, 20])
        let phrase = DetectedPhrase(
            term: term,
            startTime: 1.5,
            endTime: 2.0,
            confidence: 0.95,
            wasBoosted: true
        )

        XCTAssertEqual(phrase.term.text, "NVIDIA")
        XCTAssertEqual(phrase.startTime, 1.5)
        XCTAssertEqual(phrase.endTime, 2.0)
        XCTAssertEqual(phrase.confidence, 0.95)
        XCTAssertTrue(phrase.wasBoosted)
    }

    func testDetectedPhraseCodable() throws {
        let term = CustomVocabularyTerm(text: "hello", tokenIds: [1, 2])
        let phrase = DetectedPhrase(
            term: term,
            startTime: 0.5,
            endTime: 1.0,
            confidence: 0.9,
            wasBoosted: false
        )

        let data = try JSONEncoder().encode(phrase)
        let decoded = try JSONDecoder().decode(DetectedPhrase.self, from: data)
        XCTAssertEqual(decoded.term.text, "hello")
        XCTAssertEqual(decoded.startTime, 0.5)
        XCTAssertFalse(decoded.wasBoosted)
    }

    // MARK: - ASRResult Integration

    func testASRResultWithDetectedPhrases() {
        let term = CustomVocabularyTerm(text: "test", tokenIds: [1])
        let phrase = DetectedPhrase(
            term: term, startTime: 0.0, endTime: 0.5, confidence: 0.9, wasBoosted: true)
        let result = ASRResult(
            text: "test",
            confidence: 0.9,
            duration: 1.0,
            processingTime: 0.1,
            detectedPhrases: [phrase]
        )

        XCTAssertEqual(result.detectedPhrases?.count, 1)
        XCTAssertEqual(result.detectedPhrases?.first?.term.text, "test")
    }

    func testASRResultWithoutDetectedPhrases() {
        let result = ASRResult(
            text: "hello",
            confidence: 0.8,
            duration: 1.0,
            processingTime: 0.1
        )
        XCTAssertNil(result.detectedPhrases)
    }

    func testASRResultRescoringPreservesDetectedPhrases() {
        let term = CustomVocabularyTerm(text: "keyword", tokenIds: [1])
        let phrase = DetectedPhrase(
            term: term, startTime: 0.0, endTime: 0.5, confidence: 0.9, wasBoosted: true)
        let original = ASRResult(
            text: "original",
            confidence: 0.9,
            duration: 1.0,
            processingTime: 0.1,
            detectedPhrases: [phrase]
        )

        let rescored = original.withRescoring(text: "rescored", detected: ["x"], applied: ["x"])
        XCTAssertEqual(rescored.detectedPhrases?.count, 1)
        XCTAssertEqual(rescored.detectedPhrases?.first?.term.text, "keyword")
    }
}
