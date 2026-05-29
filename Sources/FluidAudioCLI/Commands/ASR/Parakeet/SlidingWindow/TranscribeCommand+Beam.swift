import CoreML
import Foundation

import FluidAudio

/// Result of a beam-mode transcription. Mirrors the public `ASRResult`
/// shape so the rest of the CLI flow can treat it uniformly.
struct BeamTranscribeResult {
    let text: String
    let tokens: [Int]
    let timestamps: [Int]
    let confidences: [Float]
    let processingTime: TimeInterval
}

/// Run TDT beam search end-to-end on an audio buffer.
///
/// Audio longer than the encoder window is decoded in overlapping chunks,
/// then token-level suffix/prefix deduplication stitches the chunk outputs.
func runTdtBeamTranscribe(
    asrModels: AsrModels?,
    audioSamples: [Float],
    beamConfig: TdtBeamConfig,
    logger: AppLogger
) async throws -> BeamTranscribeResult {
    guard let asrModels else {
        throw ASRError.notInitialized
    }
    guard let jointLogits = asrModels.jointLogits else {
        throw ASRError.processingFailed(
            "Beam mode requires JointDecisionLogits.mlmodelc to be loaded"
        )
    }
    let maxSamples = ASRConstants.maxModelSamples

    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")

    // Reuse TdtRescorer's encoder runner — it owns the pre-processor +
    // encoder calls and returns the right MLMultiArray shape. We don't
    // need any of the rescoring logic here.
    let encoderRunner = try TdtRescorer(
        asrModels: asrModels, tokenizerModelURL: tokenizerURL)

    let t0 = Date()
    let chunkStarts = beamChunkStarts(audioSampleCount: audioSamples.count, maxSamples: maxSamples)
    logger.info("Beam: \(chunkStarts.count) chunk(s) for \(audioSamples.count) samples")

    // Build the decoder with the right blankId for this model version.
    let tdtConfig = TdtConfig(blankId: asrModels.version.blankId)
    let asrConfig = ASRConfig(
        tdtConfig: tdtConfig,
        encoderHiddenSize: asrModels.version.encoderHiddenSize
    )
    let decoder = TdtBeamDecoder(config: asrConfig, beamConfig: beamConfig)

    var previousTokens: [Int] = []
    var allTokens: [Int] = []
    var allTimestamps: [Int] = []
    var allConfidences: [Float] = []
    var totalScore: Float = 0
    for (chunkIndex, start) in chunkStarts.enumerated() {
        let end = min(start + maxSamples, audioSamples.count)
        var chunkAudio = Array(audioSamples[start..<end])
        if chunkAudio.count < maxSamples {
            chunkAudio.append(contentsOf: [Float](repeating: 0, count: maxSamples - chunkAudio.count))
        }

        let (encoderOutput, validLength) = try await encoderRunner.runEncoder(audioSamples: chunkAudio)
        let initialState = TdtDecoderState.make(decoderLayers: asrModels.version.decoderLayers)
        let frameOffset = start / ASRConstants.samplesPerEncoderFrame
        let result = try await decoder.decode(
            encoderOutput: encoderOutput,
            encoderSequenceLength: validLength,
            decoderModel: asrModels.decoder,
            jointLogitsModel: jointLogits,
            initialState: initialState,
            globalFrameOffset: frameOffset,
            isLastChunk: chunkIndex == chunkStarts.count - 1
        )

        let lastEmittedFrame = allTimestamps.last ?? Int.min
        var removedCount = chunkIndex == 0 ? 0 : result.timestamps.prefix { $0 <= lastEmittedFrame + 1 }.count
        removedCount += overlapPrefixLength(
            previous: previousTokens,
            current: Array(result.tokens.dropFirst(removedCount))
        )
        let keptTokens = Array(result.tokens.dropFirst(removedCount))
        allTokens.append(contentsOf: keptTokens)
        allTimestamps.append(contentsOf: result.timestamps.dropFirst(removedCount))
        allConfidences.append(contentsOf: result.tokenConfidences.dropFirst(removedCount))
        previousTokens = allTokens
        totalScore += result.score
    }
    let processingTime = Date().timeIntervalSince(t0)

    // Detokenize the token sequence into text. Use the TDT vocabulary
    // (which was loaded with the model) and strip SentencePiece markers.
    let text = detokenize(
        tokenIds: allTokens,
        vocab: asrModels.vocabulary
    )

    logger.info(
        "Beam: emitted \(allTokens.count) tokens, score=\(String(format: "%.2f", totalScore)) in \(String(format: "%.3f", processingTime))s"
    )

    return BeamTranscribeResult(
        text: text,
        tokens: allTokens,
        timestamps: allTimestamps,
        confidences: allConfidences,
        processingTime: processingTime
    )
}

private func beamChunkStarts(audioSampleCount: Int, maxSamples: Int) -> [Int] {
    guard audioSampleCount > maxSamples else { return [0] }
    let overlapSamples = 2 * ASRConstants.sampleRate
    let stride = max(ASRConstants.samplesPerEncoderFrame, maxSamples - overlapSamples)
    var starts: [Int] = []
    var start = 0
    while start < audioSampleCount {
        starts.append(start)
        if start + maxSamples >= audioSampleCount { break }
        start += stride
    }
    return starts
}

private func overlapPrefixLength(previous: [Int], current: [Int], maxOverlap: Int = 24) -> Int {
    guard !previous.isEmpty, !current.isEmpty else { return 0 }
    let limit = min(maxOverlap, previous.count, current.count)
    if limit == 0 { return 0 }
    for length in stride(from: limit, through: 1, by: -1) {
        if Array(previous.suffix(length)) == Array(current.prefix(length)) {
            return length
        }
    }
    return 0
}

/// Tokenize a custom vocabulary against the TDT vocab (SentencePiece
/// Unigram via tokenizer.model) for use as bias keywords.
func tokenizeVocabularyForBeamBias(
    asrModels: AsrModels,
    vocabulary: CustomVocabularyContext
) throws -> [[Int]] {
    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")
    guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
        throw ASRError.processingFailed(
            "Beam bias requires tokenizer.model at \(tokenizerURL.path)"
        )
    }
    let data = try Data(contentsOf: tokenizerURL)
    let tokenizer = try SentencePieceTokenizer(modelData: data)
    return vocabulary.terms.compactMap { term in
        let ids = tokenizer.encode(term.text)
        return ids.isEmpty ? nil : ids
    }
}

// MARK: - Detokenization

/// Decode a token-ID sequence into a UTF-8 string using the TDT vocab.
/// Strips the SentencePiece word-boundary marker (`▁`) and inserts spaces
/// at word boundaries. Matches the behavior of the existing greedy path's
/// downstream string assembly.
private func detokenize(tokenIds: [Int], vocab: [Int: String]) -> String {
    var pieces: [String] = []
    pieces.reserveCapacity(tokenIds.count)
    for id in tokenIds {
        guard let piece = vocab[id] else { continue }
        pieces.append(piece)
    }
    let joined = pieces.joined()
    return joined.replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: " ")
        .trimmingCharacters(in: .whitespaces)
}
