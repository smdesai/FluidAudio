import XCTest

@testable import FluidAudioCLI

#if os(macOS)
final class CtcEarningsBenchmarkTests: XCTestCase {

    func testDistractorFalseAcceptCountsActiveVocabTermsNotInReferenceOrCheckWords() {
        let report = CtcEarningsBenchmark.computeDistractorFalseAccepts(
            vocabularyWords: ["Hemacord", "Priorix", "Evenity"],
            checkWords: ["Hemacord"],
            referenceNormalized: "the patient received hemacord today",
            hypothesisNormalized: "the patient received hemacord and priorix today"
        )

        XCTAssertEqual(report.count, 1)
        XCTAssertEqual(report.terms, ["Priorix"])
    }

    func testDistractorFalseAcceptIgnoresVocabularyTermsPresentInReference() {
        let report = CtcEarningsBenchmark.computeDistractorFalseAccepts(
            vocabularyWords: ["Priorix"],
            checkWords: [],
            referenceNormalized: "priorix was discussed",
            hypothesisNormalized: "priorix was discussed"
        )

        XCTAssertEqual(report.count, 0)
        XCTAssertTrue(report.terms.isEmpty)
    }

    func testDistractorFalseAcceptUsesWholeNormalizedPhraseMatching() {
        let report = CtcEarningsBenchmark.computeDistractorFalseAccepts(
            vocabularyWords: ["New Drug", "Priorix"],
            checkWords: [],
            referenceNormalized: "we reviewed the prior plan",
            hypothesisNormalized: "we reviewed the prior plan for a new drug application"
        )

        XCTAssertEqual(report.count, 1)
        XCTAssertEqual(report.terms, ["New Drug"])
    }
}
#endif
