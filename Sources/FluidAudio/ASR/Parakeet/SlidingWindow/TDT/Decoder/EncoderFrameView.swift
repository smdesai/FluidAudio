import Accelerate
import CoreML
import Foundation

/// Lightweight view over encoder frames that preserves original strides for zero-copy access.
/// Provides contiguous frame vectors on demand without materializing intermediate arrays.
struct EncoderFrameView {
    let hiddenSize: Int
    let count: Int

    private let array: MLMultiArray
    private let timeAxis: Int
    private let hiddenAxis: Int
    private let timeStride: Int
    private let hiddenStride: Int
    private let timeBaseOffset: Int
    private let baseFloat32Pointer: UnsafeMutablePointer<Float>?
    private let baseFloat16Pointer: UnsafeMutablePointer<UInt16>?

    /// Initialize with explicit hidden size (for model-version-aware callers)
    init(encoderOutput: MLMultiArray, validLength: Int, expectedHiddenSize: Int) throws {
        let shape = encoderOutput.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(shape)")
        }
        guard shape[0] == 1 else {
            throw ASRError.processingFailed("Unsupported batch dimension: \(shape[0])")
        }

        let hiddenSize = expectedHiddenSize
        let axis1MatchesHidden = shape[1] == hiddenSize
        let axis2MatchesHidden = shape[2] == hiddenSize
        guard axis1MatchesHidden || axis2MatchesHidden else {
            throw ASRError.processingFailed("Encoder hidden size mismatch: \(shape), expected \(hiddenSize)")
        }

        self.hiddenAxis = axis1MatchesHidden ? 1 : 2
        self.timeAxis = axis1MatchesHidden ? 2 : 1
        self.hiddenSize = hiddenSize

        let strides = encoderOutput.strides.map { $0.intValue }
        self.hiddenStride = strides[self.hiddenAxis]
        self.timeStride = strides[self.timeAxis]

        let availableFrames = shape[self.timeAxis]
        self.count = min(validLength, availableFrames)
        guard count > 0 else {
            throw ASRError.processingFailed("Encoder output has no frames")
        }
        self.array = encoderOutput

        switch encoderOutput.dataType {
        case .float32:
            self.baseFloat32Pointer = encoderOutput.dataPointer.bindMemory(
                to: Float.self, capacity: encoderOutput.count)
            self.baseFloat16Pointer = nil
        case .float16:
            self.baseFloat32Pointer = nil
            self.baseFloat16Pointer = encoderOutput.dataPointer.bindMemory(
                to: UInt16.self, capacity: encoderOutput.count)
        default:
            throw ASRError.processingFailed("Unsupported encoder output type: \(encoderOutput.dataType)")
        }

        if timeStride >= 0 {
            self.timeBaseOffset = 0
        } else {
            self.timeBaseOffset = (availableFrames - 1) * timeStride
        }
    }

    /// Convenience initializer using default encoder hidden size from ASRConstants
    init(encoderOutput: MLMultiArray, validLength: Int) throws {
        try self.init(
            encoderOutput: encoderOutput,
            validLength: validLength,
            expectedHiddenSize: ASRConstants.encoderHiddenSize
        )
    }

    func copyFrame(
        at index: Int,
        into destination: UnsafeMutablePointer<Float>,
        destinationStride: Int
    ) throws {
        guard index >= 0 && index < count else {
            throw ASRError.processingFailed("Encoder frame index out of range: \(index)")
        }

        let frameOffset = timeBaseOffset + index * timeStride

        guard hiddenStride != 0 else {
            throw ASRError.processingFailed("Invalid hidden stride: 0")
        }

        if let baseFloat32Pointer {
            let frameStart = baseFloat32Pointer.advanced(by: frameOffset)
            let sourcePointer = UnsafePointer<Float>(frameStart)
            let count = try makeBlasIndex(hiddenSize, label: "Hidden size")
            let incX = try makeBlasIndex(hiddenStride, label: "Hidden stride")
            let destStrideCblas = try makeBlasIndex(destinationStride, label: "Destination stride")

            if hiddenStride == 1 && destinationStride == 1 {
                destination.update(from: sourcePointer, count: hiddenSize)
            } else {
                cblas_scopy(count, sourcePointer, incX, destination, destStrideCblas)
            }
            return
        }

        guard let baseFloat16Pointer else {
            throw ASRError.processingFailed("Encoder output has no readable backing pointer")
        }

        if hiddenStride == 1 && destinationStride == 1 {
            let sourcePointer = UnsafePointer<UInt16>(baseFloat16Pointer.advanced(by: frameOffset))
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: sourcePointer),
                height: 1,
                width: vImagePixelCount(hiddenSize),
                rowBytes: hiddenSize * MemoryLayout<UInt16>.stride
            )
            var dst = vImage_Buffer(
                data: destination,
                height: 1,
                width: vImagePixelCount(hiddenSize),
                rowBytes: hiddenSize * MemoryLayout<Float>.stride
            )
            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        } else {
            var packed = [UInt16](repeating: 0, count: hiddenSize)
            let sourceBase = baseFloat16Pointer.advanced(by: frameOffset)
            for hiddenIndex in 0..<hiddenSize {
                packed[hiddenIndex] = sourceBase[hiddenIndex * hiddenStride]
            }
            var converted = [Float](repeating: 0, count: hiddenSize)
            packed.withUnsafeBufferPointer { packedBuffer in
                converted.withUnsafeMutableBufferPointer { convertedBuffer in
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: packedBuffer.baseAddress!),
                        height: 1,
                        width: vImagePixelCount(hiddenSize),
                        rowBytes: hiddenSize * MemoryLayout<UInt16>.stride
                    )
                    var dst = vImage_Buffer(
                        data: convertedBuffer.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(hiddenSize),
                        rowBytes: hiddenSize * MemoryLayout<Float>.stride
                    )
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            for hiddenIndex in 0..<hiddenSize {
                destination[hiddenIndex * destinationStride] = converted[hiddenIndex]
            }
        }
    }
}
