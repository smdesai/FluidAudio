import XCTest

@testable import FluidAudio

final class CtcWordAlignmentTests: XCTestCase {

    func testAlignmentBuildsWordsAndFrameRanges() {
        let vocabulary = [
            0: "▁hello",
            1: "wor",
            2: "ld",
        ]
        let blankId = 3
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0, 0.0],
            [-100.0, 0.0, -100.0, -100.0],
            [-100.0, -100.0, 0.0, -100.0],
        ]

        let alignment = CtcWordAligner.align(
            logProbs: logProbs,
            vocabulary: vocabulary,
            blankId: blankId,
            frameDuration: 0.04,
            tokenWeight: 0.5
        )

        XCTAssertEqual(alignment.count, 1)
        XCTAssertEqual(alignment[0].word, "helloworld")
        XCTAssertEqual(alignment[0].tokenIds, [0, 1, 2])
        XCTAssertEqual(alignment[0].startFrame, 0)
        XCTAssertEqual(alignment[0].endFrame, 3)
        XCTAssertEqual(alignment[0].score, 1.5, accuracy: 0.001)
        XCTAssertEqual(alignment[0].startTime, 0.0, accuracy: 0.001)
        XCTAssertEqual(alignment[0].endTime, 0.12, accuracy: 0.001)
    }

    func testAlignmentSplitsOnWordBoundaryTokens() {
        let vocabulary = [
            0: "▁hello",
            1: "▁world",
        ]
        let blankId = 2
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0],
            [-100.0, -100.0, 0.0],
            [-100.0, 0.0, -100.0],
        ]

        let alignment = CtcWordAligner.align(
            logProbs: logProbs,
            vocabulary: vocabulary,
            blankId: blankId,
            frameDuration: 0.08,
            tokenWeight: 0.5
        )

        XCTAssertEqual(alignment.map { $0.word }, ["hello", "world"])
        XCTAssertEqual(alignment.map { $0.startFrame }, [0, 2])
        XCTAssertEqual(alignment.map { $0.endFrame }, [0, 2])
    }

    func testAlignmentCollapsesRepeatsUnlessSeparatedByBlank() {
        let vocabulary = [0: "▁ha"]
        let blankId = 1
        let logProbs: [[Float]] = [
            [0.0, -100.0],
            [0.0, -100.0],
            [-100.0, 0.0],
            [0.0, -100.0],
        ]

        let alignment = CtcWordAligner.align(
            logProbs: logProbs,
            vocabulary: vocabulary,
            blankId: blankId,
            frameDuration: 0.04,
            tokenWeight: 0
        )

        XCTAssertEqual(alignment.map { $0.word }, ["ha", "ha"])
        XCTAssertEqual(alignment.map { $0.startFrame }, [0, 3])
    }

    func testAlignmentValidatorRejectsCandidateBelowGreedyAlignment() {
        let alignment = CtcWordAlignment(
            word: "cloud",
            tokenIds: [0, 1],
            score: 1.0,
            startFrame: 10,
            endFrame: 12,
            startTime: 0.4,
            endTime: 0.48
        )

        let passes = CtcAlignmentValidator.candidateBeatsGreedyAlignment(
            candidateScore: 0.49,
            candidateStartFrame: 11,
            candidateEndFrame: 13,
            alignments: [alignment]
        )

        XCTAssertFalse(passes)
    }

    func testAlignmentValidatorAcceptsCandidateAboveGreedyAlignment() {
        let alignment = CtcWordAlignment(
            word: "cloud",
            tokenIds: [0, 1],
            score: 1.0,
            startFrame: 10,
            endFrame: 12,
            startTime: 0.4,
            endTime: 0.48
        )

        let passes = CtcAlignmentValidator.candidateBeatsGreedyAlignment(
            candidateScore: 0.51,
            candidateStartFrame: 11,
            candidateEndFrame: 13,
            alignments: [alignment]
        )

        XCTAssertTrue(passes)
    }

    func testAlignmentValidatorAcceptsWhenNoGreedyOverlap() {
        let alignment = CtcWordAlignment(
            word: "cloud",
            tokenIds: [0, 1],
            score: 100.0,
            startFrame: 10,
            endFrame: 12,
            startTime: 0.4,
            endTime: 0.48
        )

        let passes = CtcAlignmentValidator.candidateBeatsGreedyAlignment(
            candidateScore: -100.0,
            candidateStartFrame: 20,
            candidateEndFrame: 22,
            alignments: [alignment]
        )

        XCTAssertTrue(passes)
    }

    // MARK: - Large-vocab false-accept veto (fix #1)

    /// The `prior` → `priorix` regression: the greedy CTC decode produced the
    /// correct common word `prior` with a strong per-token score; the distractor
    /// drug name `priorix` only wins the CTC-vs-CTC comparison because of the
    /// +cbw boost. With a large vocabulary the alignment veto must block it.
    private func priorAlignment() -> CtcWordAlignment {
        // Greedy "prior": 2 tokens, summed score -0.6 → normalizedScore -0.3.
        CtcWordAlignment(
            word: "prior",
            tokenIds: [11, 12],
            score: -0.6,
            startFrame: 40,
            endFrame: 43,
            startTime: 1.6,
            endTime: 1.72
        )
    }

    func testLargeVocabVetoBlocksBoostedDistractorBelowGreedyWord() {
        // priorix boosted score -0.5 is below the greedy "prior" word (-0.3),
        // so even after the +cbw boost the replacement must be vetoed.
        let passes = CtcAlignmentValidator.candidatePassesLargeVocabAlignmentVeto(
            boostedVocabScore: -0.5,
            candidateStartFrame: 41,
            candidateEndFrame: 44,
            alignments: [priorAlignment()],
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertFalse(passes, "Boosted distractor below the overlapping greedy word must be vetoed in large vocab")
    }

    func testLargeVocabVetoDisabledForSmallVocab() {
        // Identical scores, but a small vocabulary leaves the veto disabled so
        // the proven small-dictionary path (100% precision) is unchanged.
        let passes = CtcAlignmentValidator.candidatePassesLargeVocabAlignmentVeto(
            boostedVocabScore: -0.5,
            candidateStartFrame: 41,
            candidateEndFrame: 44,
            alignments: [priorAlignment()],
            vocabularyTermCount: 3,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "Veto must not fire at or below the large-vocab threshold")
    }

    func testLargeVocabVetoAllowsCandidateBeatingGreedyWord() {
        // A genuine keyword whose boosted score beats the greedy word is allowed
        // through even in a large vocabulary.
        let passes = CtcAlignmentValidator.candidatePassesLargeVocabAlignmentVeto(
            boostedVocabScore: -0.1,
            candidateStartFrame: 41,
            candidateEndFrame: 44,
            alignments: [priorAlignment()],
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "Candidate beating the greedy word must still be accepted")
    }

    func testLargeVocabVetoAllowsWhenNoGreedyOverlap() {
        // No greedy word covers the candidate frames → nothing to defend, allow.
        let passes = CtcAlignmentValidator.candidatePassesLargeVocabAlignmentVeto(
            boostedVocabScore: -50.0,
            candidateStartFrame: 200,
            candidateEndFrame: 204,
            alignments: [priorAlignment()],
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "No overlapping greedy word means no veto")
    }

    // MARK: - Raw-acoustic-margin gate (fix #2)

    /// The +cbw boost can flip the CTC-vs-CTC comparison even when the
    /// distractor's *raw* (pre-boost) acoustic score is far below the original
    /// word's. The margin gate requires the distractor to have real acoustic
    /// support — its raw score must not trail the original by more than `slack`.
    func testRawAcousticMarginVetoesDistractorWithNoAcousticSupport() {
        // Distractor raw score -6.0 trails the original word -1.0 by 5.0,
        // well beyond a 1.5 slack: only the boost carried it, so veto.
        let passes = CtcAlignmentValidator.passesLargeVocabRawAcousticMargin(
            rawVocabScore: -6.0,
            originalScore: -1.0,
            slack: 1.5,
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertFalse(passes, "Distractor with no raw acoustic support must be vetoed")
    }

    func testRawAcousticMarginAllowsDistractorWithinSlack() {
        // Raw score -2.0 trails original -1.0 by only 1.0 (<= slack 1.5): the
        // distractor has genuine acoustic support, so allow the boost to decide.
        let passes = CtcAlignmentValidator.passesLargeVocabRawAcousticMargin(
            rawVocabScore: -2.0,
            originalScore: -1.0,
            slack: 1.5,
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "Distractor with raw support within slack must pass")
    }

    func testRawAcousticMarginAllowsWhenVocabBeatsOriginalRaw() {
        // A genuine keyword whose raw score already beats the original passes.
        let passes = CtcAlignmentValidator.passesLargeVocabRawAcousticMargin(
            rawVocabScore: -0.5,
            originalScore: -1.0,
            slack: 1.5,
            vocabularyTermCount: 650,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "Keyword beating the original raw score must pass")
    }

    func testRawAcousticMarginDisabledForSmallVocab() {
        // Same large trailing margin, but a small vocabulary leaves the gate
        // disabled so the proven small-dictionary path is unchanged.
        let passes = CtcAlignmentValidator.passesLargeVocabRawAcousticMargin(
            rawVocabScore: -6.0,
            originalScore: -1.0,
            slack: 1.5,
            vocabularyTermCount: 3,
            largeVocabThreshold: 10
        )
        XCTAssertTrue(passes, "Margin gate must not fire at or below the large-vocab threshold")
    }
}
