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
}
