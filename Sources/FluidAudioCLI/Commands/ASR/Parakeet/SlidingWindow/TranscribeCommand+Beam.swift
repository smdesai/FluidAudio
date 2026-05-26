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

/// Run TDT beam search end-to-end on a single audio buffer.
///
/// Bypasses the sliding-window greedy pipeline; runs the encoder once over
/// the full audio (padded to the 15s window) and decodes with
/// `TdtBeamDecoder`. When `keywordTokenSequences` is non-empty, shallow-
/// fusion biasing is enabled inside the beam decoder.
///
/// Throws when:
/// - The loaded model bundle doesn't expose `jointLogits`.
/// - Audio exceeds the encoder's 15s window (caller should fall back).
func runTdtBeamTranscribe(
    asrModels: AsrModels?,
    audioSamples: [Float],
    vocabulary: CustomVocabularyContext?,
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
    guard audioSamples.count <= maxSamples else {
        throw ASRError.processingFailed(
            "Beam mode v1 requires audio ≤ \(maxSamples) samples; got \(audioSamples.count). "
                + "Long-audio sliding-window beam decoding not implemented yet."
        )
    }

    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")

    // Reuse TdtRescorer's encoder runner — it owns the pre-processor +
    // encoder calls and returns the right MLMultiArray shape. We don't
    // need any of the rescoring logic here.
    let encoderRunner = try TdtRescorer(
        asrModels: asrModels, tokenizerModelURL: tokenizerURL)

    var padded = audioSamples
    if padded.count < maxSamples {
        padded.append(contentsOf: [Float](repeating: 0, count: maxSamples - padded.count))
    }

    let t0 = Date()
    let (encoderOutput, validLength) = try await encoderRunner.runEncoder(audioSamples: padded)
    logger.info(
        "Beam: encoder ready (\(validLength) valid frames, \(audioSamples.count) audio samples)"
    )

    // Build the decoder with the right blankId for this model version.
    let tdtConfig = TdtConfig(blankId: asrModels.version.blankId)
    let asrConfig = ASRConfig(
        tdtConfig: tdtConfig,
        encoderHiddenSize: asrModels.version.encoderHiddenSize
    )
    let decoder = TdtBeamDecoder(config: asrConfig, beamConfig: beamConfig)

    let initialState = TdtDecoderState.make(decoderLayers: asrModels.version.decoderLayers)
    let result: TdtBeamDecoder.DecodeResult = try await decoder.decode(
        encoderOutput: encoderOutput,
        encoderSequenceLength: validLength,
        decoderModel: asrModels.decoder,
        jointLogitsModel: jointLogits,
        initialState: initialState
    )
    let processingTime = Date().timeIntervalSince(t0)

    // Detokenize the token sequence into text. Use the TDT vocabulary
    // (which was loaded with the model) and strip SentencePiece markers.
    let text = detokenize(
        tokenIds: result.tokens,
        vocab: asrModels.vocabulary
    )

    logger.info(
        "Beam: emitted \(result.tokens.count) tokens, score=\(String(format: "%.2f", result.score)) in \(String(format: "%.3f", processingTime))s"
    )

    return BeamTranscribeResult(
        text: text,
        tokens: result.tokens,
        timestamps: result.timestamps,
        confidences: result.tokenConfidences,
        processingTime: processingTime
    )
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
