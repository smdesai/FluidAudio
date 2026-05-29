import Accelerate
import CoreML
import Foundation

enum CtcLogProbUtils {

    static func logProbs(
        from logits: MLMultiArray,
        blankId: Int,
        temperature: Float = ContextBiasingConstants.ctcTemperature,
        blankBias: Float = ContextBiasingConstants.blankBias,
        validFrames: Int? = nil
    ) throws -> [[Float]] {
        let rows = try rawRows(from: logits)
        let trimmedRows: [[Float]]
        if let validFrames {
            trimmedRows = Array(rows.prefix(max(0, min(validFrames, rows.count))))
        } else {
            trimmedRows = rows
        }
        return CtcKeywordSpotter.applyLogSoftmax(
            rawLogits: trimmedRows,
            blankId: blankId,
            temperature: temperature,
            blankBias: blankBias
        )
    }

    private static func rawRows(from logits: MLMultiArray) throws -> [[Float]] {
        let rank = logits.shape.count
        guard rank == 3 || rank == 4 else {
            throw ASRError.processingFailed("Unexpected CTC output rank: \(logits.shape)")
        }

        let shape = logits.shape.map { $0.intValue }
        let strides = logits.strides.map { $0.intValue }

        let vocabSize: Int
        let timeSteps: Int
        let timeStride: Int
        let vocabStride: Int

        if rank == 3 {
            timeSteps = shape[1]
            vocabSize = shape[2]
            timeStride = strides[1]
            vocabStride = strides[2]
        } else {
            vocabSize = shape[1]
            timeSteps = shape[3]
            vocabStride = strides[1]
            timeStride = strides[3]
        }

        guard vocabSize > 0, timeSteps > 0 else { return [] }

        switch logits.dataType {
        case .float32:
            return rawRowsFloat32(
                logits: logits,
                timeSteps: timeSteps,
                vocabSize: vocabSize,
                timeStride: timeStride,
                vocabStride: vocabStride
            )
        case .float16:
            return rawRowsFloat16(
                logits: logits,
                timeSteps: timeSteps,
                vocabSize: vocabSize,
                timeStride: timeStride,
                vocabStride: vocabStride
            )
        default:
            throw ASRError.processingFailed("Unsupported CTC output dtype: \(logits.dataType.rawValue)")
        }
    }

    private static func rawRowsFloat32(
        logits: MLMultiArray,
        timeSteps: Int,
        vocabSize: Int,
        timeStride: Int,
        vocabStride: Int
    ) -> [[Float]] {
        let basePtr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        var rows: [[Float]] = []
        rows.reserveCapacity(timeSteps)
        for t in 0..<timeSteps {
            var row = [Float](repeating: 0, count: vocabSize)
            row.withUnsafeMutableBufferPointer { dst in
                let dstPtr = dst.baseAddress!
                if vocabStride == 1 {
                    dstPtr.update(from: basePtr.advanced(by: t * timeStride), count: vocabSize)
                } else {
                    var src = basePtr.advanced(by: t * timeStride)
                    for v in 0..<vocabSize {
                        dstPtr[v] = src.pointee
                        src = src.advanced(by: vocabStride)
                    }
                }
            }
            rows.append(row)
        }
        return rows
    }

    private static func rawRowsFloat16(
        logits: MLMultiArray,
        timeSteps: Int,
        vocabSize: Int,
        timeStride: Int,
        vocabStride: Int
    ) -> [[Float]] {
        let basePtr = logits.dataPointer.bindMemory(to: UInt16.self, capacity: logits.count)
        var rows: [[Float]] = []
        rows.reserveCapacity(timeSteps)

        var fp16Row = [UInt16](repeating: 0, count: vocabSize)
        for t in 0..<timeSteps {
            let srcStart = basePtr.advanced(by: t * timeStride)
            fp16Row.withUnsafeMutableBufferPointer { fp16Buf in
                let fp16Ptr = fp16Buf.baseAddress!
                if vocabStride == 1 {
                    fp16Ptr.update(from: srcStart, count: vocabSize)
                } else {
                    var src = srcStart
                    for v in 0..<vocabSize {
                        fp16Ptr[v] = src.pointee
                        src = src.advanced(by: vocabStride)
                    }
                }
            }

            var row = [Float](repeating: 0, count: vocabSize)
            fp16Row.withUnsafeMutableBufferPointer { fp16Buf in
                row.withUnsafeMutableBufferPointer { fp32Buf in
                    var src = vImage_Buffer(
                        data: fp16Buf.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(vocabSize),
                        rowBytes: vocabSize * MemoryLayout<UInt16>.stride
                    )
                    var dst = vImage_Buffer(
                        data: fp32Buf.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(vocabSize),
                        rowBytes: vocabSize * MemoryLayout<Float>.stride
                    )
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            rows.append(row)
        }
        return rows
    }
}
