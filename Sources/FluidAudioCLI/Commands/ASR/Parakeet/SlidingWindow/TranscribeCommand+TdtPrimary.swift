import CoreML
import Foundation

import FluidAudio

/// Build a `TdtScorerContext` for the rescorer's primary-scorer mode.
///
/// Strategy:
/// - Audio ≤ 15s: one encoder run covers the whole utterance.
/// - Audio > 15s: tile the audio with overlapping 15s windows so every
///   phrase falls inside at least one run. Stride is 12s (3s overlap)
///   which keeps any single ≤3s phrase from straddling windows.
///
/// Throws when:
/// - `asrModels` is nil or `jointLogits` is missing.
/// - The tokenizer.model isn't installed alongside the v2 models.
/// - The encoder fails on any window.
func buildTdtScorerContext(
    asrModels: AsrModels?,
    audioSamples: [Float],
    tokenTimings: [TokenTiming],
    acceptMargin: Float,
    minCandidateScore: Float,
    logger: AppLogger
) async throws -> VocabularyRescorer.TdtScorerContext {
    guard let asrModels else {
        throw ASRError.notInitialized
    }
    guard asrModels.jointLogits != nil else {
        throw ASRError.processingFailed(
            "TDT primary scoring requires JointDecisionLogits.mlmodelc to be loaded"
        )
    }

    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")
    guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
        throw ASRError.processingFailed(
            "TDT primary scoring requires tokenizer.model at \(tokenizerURL.path)"
        )
    }

    let scorer = try TdtRescorer(
        asrModels: asrModels, tokenizerModelURL: tokenizerURL)

    let maxSamples = ASRConstants.maxModelSamples
    // Stride = 12s (75% of window). Any phrase ≤ 3s is fully contained in
    // at least one window even when straddling a stride boundary.
    let strideSamples = Int(Double(maxSamples) * 0.8)

    var runs: [VocabularyRescorer.TdtEncoderRun] = []
    var windowStart = 0
    while windowStart < audioSamples.count {
        let windowEnd = min(audioSamples.count, windowStart + maxSamples)
        var slice = Array(audioSamples[windowStart..<windowEnd])
        if slice.count < maxSamples {
            slice.append(contentsOf: [Float](repeating: 0, count: maxSamples - slice.count))
        }
        let (encoder, validLength) = try await scorer.runEncoder(audioSamples: slice)
        runs.append(
            VocabularyRescorer.TdtEncoderRun(
                encoder: encoder,
                validLength: validLength,
                sampleStart: windowStart,
                sampleEnd: min(audioSamples.count, windowStart + maxSamples)
            )
        )
        if windowEnd >= audioSamples.count { break }
        windowStart += strideSamples
    }

    logger.info(
        "TDT primary scorer: \(runs.count) encoder run(s) covering \(audioSamples.count) samples"
    )

    return VocabularyRescorer.TdtScorerContext(
        scorer: scorer,
        tokenTimings: tokenTimings,
        encoderRuns: runs,
        sampleRate: ASRConstants.sampleRate,
        acceptMargin: acceptMargin,
        minCandidateScore: minCandidateScore
    )
}
