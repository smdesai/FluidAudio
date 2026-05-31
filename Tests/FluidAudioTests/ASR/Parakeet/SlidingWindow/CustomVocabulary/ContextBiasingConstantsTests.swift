import XCTest

@testable import FluidAudio

final class ContextBiasingConstantsTests: XCTestCase {

    // MARK: - Token ID Constants

    func testWildcardTokenId() {
        XCTAssertEqual(ContextBiasingConstants.wildcardTokenId, -1)
    }

    func testDefaultBlankId() {
        XCTAssertEqual(ContextBiasingConstants.defaultBlankId, 1024)
    }

    // MARK: - Similarity Threshold Hierarchy

    func testSimilarityThresholdHierarchy() {
        // Thresholds should form a strict ordering from lenient to strict
        let floor = ContextBiasingConstants.minSimilarityFloor
        let defaultMin = ContextBiasingConstants.defaultMinSimilarity
        let lengthRatio = ContextBiasingConstants.lengthRatioThreshold
        let shortWord = ContextBiasingConstants.shortWordSimilarity
        let stopword = ContextBiasingConstants.stopwordSpanSimilarity

        XCTAssertLessThan(floor, defaultMin)
        XCTAssertLessThan(defaultMin, lengthRatio)
        XCTAssertLessThanOrEqual(lengthRatio, shortWord)
        XCTAssertLessThanOrEqual(shortWord, stopword)
    }

    func testAllSimilarityThresholdsInRange() {
        let thresholds: [Float] = [
            ContextBiasingConstants.minSimilarityFloor,
            ContextBiasingConstants.defaultMinSimilarity,
            ContextBiasingConstants.lengthRatioThreshold,
            ContextBiasingConstants.shortWordSimilarity,
            ContextBiasingConstants.largeVocabShortWordSimilarity,
            ContextBiasingConstants.largeVocabMultiWordToSingleWordSimilarity,
            ContextBiasingConstants.stopwordSpanSimilarity,
        ]
        for threshold in thresholds {
            XCTAssertGreaterThan(threshold, 0.0)
            XCTAssertLessThanOrEqual(threshold, 1.0)
        }
    }

    // MARK: - Context Biasing Weights

    func testCbwPositive() {
        XCTAssertGreaterThan(ContextBiasingConstants.defaultCbw, 0)
    }

    func testDefaultAlphaInRange() {
        XCTAssertGreaterThanOrEqual(ContextBiasingConstants.defaultAlpha, 0.0)
        XCTAssertLessThanOrEqual(ContextBiasingConstants.defaultAlpha, 1.0)
    }

    // MARK: - rescorerConfig(forVocabSize:)

    func testSmallVocabConfig() {
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        XCTAssertEqual(config.minSimilarity, 0.50, accuracy: 0.01)
        XCTAssertEqual(config.cbw, 4.5, accuracy: 0.01)
    }

    func testLargeVocabConfig() {
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 15)
        XCTAssertEqual(config.minSimilarity, 0.55, accuracy: 0.01)
        XCTAssertEqual(config.cbw, 4.5, accuracy: 0.01)
    }

    func testBoundaryVocabConfig() {
        // Exactly 10 = threshold, NOT large (>10 is large)
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 10)
        XCTAssertEqual(config.minSimilarity, 0.50, accuracy: 0.01)
    }

    func testLargeVocabStricterThresholds() {
        let small = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        let large = ContextBiasingConstants.rescorerConfig(forVocabSize: 15)
        XCTAssertGreaterThan(large.minSimilarity, small.minSimilarity)
    }

    func testExtraLargeVocabConfig() {
        // V > 100 = extra-large, tighter similarity to suppress
        // distractor false positives.
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 500)
        XCTAssertEqual(config.minSimilarity, 0.60, accuracy: 0.01)
        XCTAssertEqual(config.cbw, 4.5, accuracy: 0.01)
    }

    func testThresholdsAreMonotoneInVocabSize() {
        // Similarity threshold must not decrease as vocab grows.
        let small = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        let large = ContextBiasingConstants.rescorerConfig(forVocabSize: 50)
        let xLarge = ContextBiasingConstants.rescorerConfig(forVocabSize: 500)
        XCTAssertLessThanOrEqual(small.minSimilarity, large.minSimilarity)
        XCTAssertLessThanOrEqual(large.minSimilarity, xLarge.minSimilarity)
    }

    // MARK: - Effective minSimilarity (context override)

    func testEffectiveMinSimilarityRespectsCallerThreshold() {
        // When a caller sets a stricter minSimilarity on CustomVocabularyContext,
        // the effective threshold should be the max of the size-based config
        // and the caller-specified value. This matches the logic in
        // AsrTranscription.applyVocabularyRescoring() and
        // SlidingWindowAsrManager.applyVocabularyRescoring().
        let smallVocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        XCTAssertEqual(smallVocabConfig.minSimilarity, 0.50, accuracy: 0.01)

        let callerThreshold: Float = 0.60
        let effective = max(smallVocabConfig.minSimilarity, callerThreshold)
        XCTAssertEqual(effective, 0.60, accuracy: 0.01, "Caller's stricter threshold should win")
    }

    func testEffectiveMinSimilarityUsesVocabConfigWhenStricter() {
        // When the size-based config is stricter than the caller's threshold,
        // the size-based config should win.
        let largeVocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: 15)
        XCTAssertEqual(largeVocabConfig.minSimilarity, 0.55, accuracy: 0.01)

        let callerThreshold: Float = 0.52
        let effective = max(largeVocabConfig.minSimilarity, callerThreshold)
        XCTAssertEqual(effective, 0.55, accuracy: 0.01, "Size-based stricter threshold should win")
    }

    // MARK: - Single-word length-tier floor (fix #3: 5-6 char dead-zone)

    private let base: Float = 0.60  // extra-large vocab floor

    func testMidLengthDeadZoneRaisedForExtraLargeVocab() {
        // 5-char common word vs 6-char distractor (first/kirsty, prior/priorix,
        // clean/creon) previously fell through every tier to the 0.60 floor.
        // The new mid-length tier must raise the bar above their similarity.
        let floor = ContextBiasingConstants.singleWordSimilarityFloor(
            wordLength: 5, vocabTermLength: 6, vocabularyTermCount: 650, base: base)
        XCTAssertGreaterThan(
            floor, 0.667, "5-char word in extra-large vocab must be gated above first->kirsty sim (0.667)")
        XCTAssertGreaterThan(floor, 0.714, "must also exceed prior->priorix sim (0.714)")
    }

    func testMidLengthTierInertForSmallVocab() {
        // Small vocab must keep the existing floor (path is 100% precision).
        let floor = ContextBiasingConstants.singleWordSimilarityFloor(
            wordLength: 5, vocabTermLength: 6, vocabularyTermCount: 5, base: 0.50)
        XCTAssertEqual(floor, 0.50, accuracy: 0.001, "small vocab unchanged")
    }

    func testShortWordTierUnchanged() {
        // <=4 char words in large vocab still use the 0.80 short-word floor.
        let floor = ContextBiasingConstants.singleWordSimilarityFloor(
            wordLength: 4, vocabTermLength: 7, vocabularyTermCount: 650, base: base)
        XCTAssertEqual(floor, ContextBiasingConstants.largeVocabShortWordSimilarity, accuracy: 0.001)
    }

    func testLongWordTierUnchanged() {
        // >=6 char words with high length ratio still use the 0.70 long-word floor.
        let floor = ContextBiasingConstants.singleWordSimilarityFloor(
            wordLength: 8, vocabTermLength: 9, vocabularyTermCount: 650, base: base)
        XCTAssertEqual(floor, ContextBiasingConstants.longWordSimilarity, accuracy: 0.001)
    }

    func testMidLengthTierConstantInRange() {
        let t = ContextBiasingConstants.extraLargeVocabMidWordSimilarity
        XCTAssertGreaterThan(t, ContextBiasingConstants.minSimilarityFloor)
        XCTAssertLessThanOrEqual(t, 1.0)
    }
}
