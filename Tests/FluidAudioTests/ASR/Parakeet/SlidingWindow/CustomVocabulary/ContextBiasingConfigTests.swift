import XCTest

@testable import FluidAudio

final class ContextBiasingConfigTests: XCTestCase {

    func testDefaultConfigUsesAutomaticCtcSource() {
        let vocab = CustomVocabularyContext(terms: [CustomVocabularyTerm(text: "NVIDIA", ctcTokenIds: [1, 2])])
        let config = ContextBiasingConfig(vocabulary: vocab)

        XCTAssertEqual(config.vocabulary.terms.count, 1)
        XCTAssertNil(config.minSimilarity)
        XCTAssertNil(config.cbw)
        XCTAssertNil(config.marginSeconds)

        switch config.ctcSource {
        case .automatic:
            break
        default:
            XCTFail("Expected automatic CTC source by default")
        }
    }

    func testSeparateCtcSourceStoresVariant() {
        let vocab = CustomVocabularyContext(terms: [])
        let config = ContextBiasingConfig(vocabulary: vocab, ctcSource: .separateCtc(.ctc06b))

        switch config.ctcSource {
        case .separateCtc(let variant):
            XCTAssertEqual(variant, .ctc06b)
        default:
            XCTFail("Expected separate CTC source")
        }
    }
}
