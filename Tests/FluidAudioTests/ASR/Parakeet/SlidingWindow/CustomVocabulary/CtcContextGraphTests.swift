import XCTest

@testable import FluidAudio

final class CtcContextGraphTests: XCTestCase {

    private func makeFrame(
        vocabSize: Int,
        hotToken: Int?,
        highScore: Float = -0.1,
        blankId: Int,
        blankScore: Float = -0.1,
        coldScore: Float = -10.0
    ) -> [Float] {
        var row = [Float](repeating: coldScore, count: vocabSize)
        if blankId < vocabSize { row[blankId] = blankScore }
        if let hotToken, hotToken < vocabSize { row[hotToken] = highScore }
        return row
    }

    func testGraphDetectsSharedPrefixTermsInSinglePass() {
        let blankId = 4
        let vocab = CustomVocabularyContext(
            terms: [
                CustomVocabularyTerm(text: "gpu", ctcTokenIds: [0, 1]),
                CustomVocabularyTerm(text: "geforce", ctcTokenIds: [0, 2]),
            ]
        )
        let graph = CtcContextGraph(
            vocabulary: vocab,
            minScore: -1.0,
            blankId: blankId
        )
        let logProbs = [
            makeFrame(vocabSize: 5, hotToken: 0, blankId: blankId),
            makeFrame(vocabSize: 5, hotToken: nil, blankId: blankId),
            makeFrame(vocabSize: 5, hotToken: 1, blankId: blankId),
            makeFrame(vocabSize: 5, hotToken: 0, blankId: blankId),
            makeFrame(vocabSize: 5, hotToken: nil, blankId: blankId),
            makeFrame(vocabSize: 5, hotToken: 2, blankId: blankId),
        ]

        let detections = graph.spot(logProbs: logProbs)
        let terms = detections.map { $0.entry.term.text }

        XCTAssertTrue(terms.contains("gpu"))
        XCTAssertTrue(terms.contains("geforce"))
        XCTAssertEqual(detections.first { $0.entry.term.text == "gpu" }?.startFrame, 0)
        XCTAssertEqual(detections.first { $0.entry.term.text == "gpu" }?.endFrame, 2)
        XCTAssertEqual(detections.first { $0.entry.term.text == "geforce" }?.startFrame, 3)
        XCTAssertEqual(detections.first { $0.entry.term.text == "geforce" }?.endFrame, 5)
    }

    func testGraphRequiresBlankBetweenRepeatedTokens() {
        let blankId = 2
        let vocab = CustomVocabularyContext(
            terms: [CustomVocabularyTerm(text: "repeat", ctcTokenIds: [0, 0])]
        )
        let graph = CtcContextGraph(vocabulary: vocab, minScore: -1.0, blankId: blankId)

        let noBlank = [
            makeFrame(vocabSize: 3, hotToken: 0, blankId: blankId, blankScore: -5.0),
            makeFrame(vocabSize: 3, hotToken: 0, blankId: blankId, blankScore: -5.0),
        ]
        let withBlank = [
            makeFrame(vocabSize: 3, hotToken: 0, blankId: blankId),
            makeFrame(vocabSize: 3, hotToken: nil, blankId: blankId),
            makeFrame(vocabSize: 3, hotToken: 0, blankId: blankId),
        ]

        XCTAssertTrue(graph.spot(logProbs: noBlank).isEmpty)
        XCTAssertEqual(graph.spot(logProbs: withBlank).count, 1)
    }

    func testGraphSkipsWildcardTermsForLegacyDpFallback() {
        let blankId = 3
        let vocab = CustomVocabularyContext(
            terms: [
                CustomVocabularyTerm(
                    text: "wild",
                    ctcTokenIds: [0, ContextBiasingConstants.wildcardTokenId, 1]
                )
            ]
        )

        let graph = CtcContextGraph(vocabulary: vocab, minScore: -1.0, blankId: blankId)

        XCTAssertTrue(graph.isEmpty)
    }
}
