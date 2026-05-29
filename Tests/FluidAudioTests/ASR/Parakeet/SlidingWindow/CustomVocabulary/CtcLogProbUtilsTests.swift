import CoreML
import XCTest

@testable import FluidAudio

final class CtcLogProbUtilsTests: XCTestCase {

    func testLogProbsFromRank3Float32() throws {
        let logits = try MLMultiArray(shape: [1, 2, 3], dataType: .float32)
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        ptr[0] = 2
        ptr[1] = 0
        ptr[2] = -1
        ptr[3] = -1
        ptr[4] = 3
        ptr[5] = 0

        let logProbs = try CtcLogProbUtils.logProbs(from: logits, blankId: 2)

        XCTAssertEqual(logProbs.count, 2)
        XCTAssertEqual(logProbs[0].count, 3)
        XCTAssertGreaterThan(logProbs[0][0], logProbs[0][1])
        XCTAssertGreaterThan(logProbs[1][1], logProbs[1][2])
    }

    func testLogProbsFromRank4Float32() throws {
        let logits = try MLMultiArray(shape: [1, 3, 1, 2], dataType: .float32)
        let shape = logits.shape.map { $0.intValue }
        let strides = logits.strides.map { $0.intValue }
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)

        func set(vocab: Int, time: Int, value: Float) {
            let index = vocab * strides[1] + time * strides[3]
            XCTAssertLessThan(index, shape.reduce(1, *))
            ptr[index] = value
        }

        for index in 0..<logits.count { ptr[index] = -5 }
        set(vocab: 0, time: 0, value: 4)
        set(vocab: 1, time: 1, value: 4)

        let logProbs = try CtcLogProbUtils.logProbs(from: logits, blankId: 2)

        XCTAssertEqual(logProbs.count, 2)
        XCTAssertGreaterThan(logProbs[0][0], logProbs[0][1])
        XCTAssertGreaterThan(logProbs[1][1], logProbs[1][0])
    }

    func testLogProbsTrimsToValidFrames() throws {
        let logits = try MLMultiArray(shape: [1, 3, 2], dataType: .float32)
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        for index in 0..<logits.count { ptr[index] = Float(index) }

        let logProbs = try CtcLogProbUtils.logProbs(from: logits, blankId: 1, validFrames: 2)

        XCTAssertEqual(logProbs.count, 2)
    }
}
