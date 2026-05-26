import CoreML
import Foundation

import FluidAudio

/// Bridge between CTC rescorer output and the experimental TDT-side
/// rescorer. Re-scores each accepted replacement using `TdtRescorer`,
/// logs the comparison, and (optionally) rebuilds the transcript with
/// only the replacements the TDT rescorer agrees with.
func applyTdtRescoreReview(
    asrModels: AsrModels?,
    audioSamples: [Float],
    originalText: String,
    rescoredText: String,
    tokenTimings: [TokenTiming],
    replacements: [VocabularyRescorer.RescoringResult],
    veto: Bool,
    vetoMargin: Float,
    vetoMinOrigScore: Float,
    logger: AppLogger
) async throws -> (
    kept: [VocabularyRescorer.RescoringResult],
    vetoed: [VocabularyRescorer.RescoringResult],
    rebuiltText: String
) {
    guard let asrModels else {
        throw ASRError.notInitialized
    }
    guard asrModels.jointLogits != nil else {
        throw ASRError.processingFailed(
            "TDT rescore requested but JointDecisionLogits.mlmodelc is not loaded"
        )
    }

    // Locate the SentencePiece tokenizer.model alongside the v2 cache.
    let tokenizerURL = AsrModels.defaultCacheDirectory(for: asrModels.version)
        .appendingPathComponent("tokenizer.model")
    guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
        throw ASRError.processingFailed(
            "TDT rescore requires tokenizer.model at \(tokenizerURL.path) (extract from .nemo)"
        )
    }

    let rescorer = try TdtRescorer(
        asrModels: asrModels, tokenizerModelURL: tokenizerURL)

    // Build the original-transcript word stream and per-word token spans so
    // we can locate each replacement's origin phrase.
    let phraseSpans = wordSpans(in: tokenTimings)

    var kept: [VocabularyRescorer.RescoringResult] = []
    var vetoed: [VocabularyRescorer.RescoringResult] = []

    let totalSamples = audioSamples.count
    let maxSamples = ASRConstants.maxModelSamples
    // Cache encoder runs keyed by window-start sample so multiple
    // replacements that fall in the same 15s window only pay the encoder
    // cost once.
    var encoderCache: [Int: (encoder: MLMultiArray, validLength: Int, sampleStart: Int)] = [:]

    for replacement in replacements where replacement.shouldReplace {
        guard let candidate = replacement.replacementWord, !candidate.isEmpty else {
            kept.append(replacement)
            continue
        }
        let originalPhrase = replacement.originalWord
        guard
            let span = locateSpan(
                phrase: originalPhrase, spans: phraseSpans, tokenTimings: tokenTimings
            )
        else {
            // Couldn't pin the phrase in the original stream — keep CTC's call.
            logger.warning(
                "TDT rescore: could not locate '\(originalPhrase)' in token timings; keeping CTC decision"
            )
            kept.append(replacement)
            continue
        }

        let phraseStartTime = tokenTimings[span.startTokenIndex].startTime
        let phraseEndTime = tokenTimings[span.startTokenIndex + span.tokenCount - 1].endTime
        let phraseStartSample = max(0, Int(phraseStartTime * Double(ASRConstants.sampleRate)))
        let phraseEndSample = min(
            totalSamples,
            Int(phraseEndTime * Double(ASRConstants.sampleRate))
        )
        let (windowStart, windowEnd) = pickWindow(
            phraseStart: phraseStartSample,
            phraseEnd: phraseEndSample,
            totalSamples: totalSamples,
            maxSamples: maxSamples
        )

        let cacheKey = windowStart
        let encoderRun: (encoder: MLMultiArray, validLength: Int, sampleStart: Int)
        if let cached = encoderCache[cacheKey] {
            encoderRun = cached
        } else {
            let slice = Array(audioSamples[windowStart..<windowEnd])
            let padded = padIfNeeded(slice, target: maxSamples)
            do {
                let (enc, len) = try await rescorer.runEncoder(audioSamples: padded)
                encoderRun = (enc, len, windowStart)
                encoderCache[cacheKey] = encoderRun
            } catch {
                logger.warning(
                    "TDT rescore encoder run failed for window [\(windowStart),\(windowEnd)): \(error.localizedDescription); keeping CTC decision"
                )
                kept.append(replacement)
                continue
            }
        }

        let shiftedTimings = shiftTimings(
            tokenTimings, sampleOffset: encoderRun.sampleStart
        )

        do {
            let decision = try rescorer.score(
                originalPhrase: originalPhrase,
                candidate: candidate,
                tokenTimings: shiftedTimings,
                phraseStartTokenIndex: span.startTokenIndex,
                phraseTokenCount: span.tokenCount,
                encoderOutput: encoderRun.encoder,
                encoderSequenceLength: encoderRun.validLength
            )

            // The replacement is "high-confidence rejected" only when:
            //   1. The original token sequence scores meaningfully better
            //      than the candidate (`margin >= vetoMargin`); AND
            //   2. TDT was confident in the original to begin with
            //      (`originalScore >= vetoMinOrigScore`).
            //
            // Condition 2 prevents the failure mode seen on FDA non-extended
            // where TDT mangles an unfamiliar drug into pseudo-English (e.g.
            // `Livten City`), then prefers its own gibberish over the
            // correct brand name. The orig score is very negative there
            // (~-35), signaling TDT itself wasn't confident.
            let margin = decision.originalScore - decision.candidateScore
            let marginPasses = margin >= vetoMargin
            let confidencePasses = decision.originalScore >= vetoMinOrigScore
            let isHighConfidenceReject =
                !decision.shouldReplace && marginPasses && confidencePasses

            let outcome: String
            if isHighConfidenceReject {
                outcome = "VETO"
            } else if decision.shouldReplace {
                outcome = "agrees"
            } else if !marginPasses {
                outcome = "below margin"
            } else {
                outcome = "below confidence"
            }
            logger.info(
                String(
                    format:
                        "  TDT-review '%@' → '%@': orig=%.2f cand=%.2f margin=%.2f → %@",
                    originalPhrase,
                    candidate,
                    decision.originalScore,
                    decision.candidateScore,
                    margin,
                    outcome
                )
            )

            if veto && isHighConfidenceReject {
                vetoed.append(replacement)
            } else {
                kept.append(replacement)
            }
        } catch {
            logger.warning(
                "TDT rescore failed for '\(originalPhrase)' → '\(candidate)': \(error.localizedDescription); keeping CTC decision"
            )
            kept.append(replacement)
        }
    }

    // Rebuild the final text by applying only `kept` replacements onto the
    // original transcript. We use simple word-level substitution; the CTC
    // rescorer already preserves casing in `replacementWord`.
    let rebuiltText: String
    if veto {
        rebuiltText = applyReplacementsToText(
            originalText: originalText, replacements: kept)
    } else {
        rebuiltText = rescoredText
    }

    return (kept: kept, vetoed: vetoed, rebuiltText: rebuiltText)
}

// MARK: - Word/phrase locator

private struct WordSpan {
    let normalizedWord: String
    let startTokenIndex: Int
    let tokenCount: Int
}

/// Reproduce VocabularyRescorer's word-boundary logic on the raw token stream
/// so we can locate phrases by exact normalized match. Returns one entry per
/// detected word in token order.
private func wordSpans(in tokenTimings: [TokenTiming]) -> [WordSpan] {
    var spans: [WordSpan] = []
    var current = ""
    var currentStart: Int = -1

    func flush(end: Int) {
        guard currentStart >= 0 else { return }
        let normalized = normalize(current)
        if !normalized.isEmpty {
            spans.append(
                WordSpan(
                    normalizedWord: normalized,
                    startTokenIndex: currentStart,
                    tokenCount: end - currentStart
                ))
        }
        current = ""
        currentStart = -1
    }

    for (idx, t) in tokenTimings.enumerated() {
        let token = t.token
        if token.isEmpty || token == "<blank>" || token == "<pad>" { continue }
        let startsWord = isWordStart(token) || currentStart < 0
        if startsWord && currentStart >= 0 {
            flush(end: idx)
        }
        if startsWord {
            current = stripBoundary(token)
            currentStart = idx
        } else {
            current += token
        }
    }
    flush(end: tokenTimings.count)
    return spans
}

private func isWordStart(_ token: String) -> Bool {
    return token.hasPrefix(ASRConstants.sentencePieceWordBoundary) || token.hasPrefix(" ")
}

private func stripBoundary(_ token: String) -> String {
    if token.hasPrefix(ASRConstants.sentencePieceWordBoundary) {
        return String(token.dropFirst(ASRConstants.sentencePieceWordBoundary.count))
    }
    if token.hasPrefix(" ") {
        return String(token.dropFirst())
    }
    return token
}

private func normalize(_ s: String) -> String {
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'-"))
    var out = ""
    for u in s.lowercased().unicodeScalars where allowed.contains(u) {
        out.append(Character(u))
    }
    return out
}

private struct LocatedSpan {
    let startTokenIndex: Int
    let tokenCount: Int
}

private func locateSpan(
    phrase: String, spans: [WordSpan], tokenTimings: [TokenTiming]
) -> LocatedSpan? {
    let phraseWords = phrase.split(separator: " ").map { normalize(String($0)) }
        .filter { !$0.isEmpty }
    guard !phraseWords.isEmpty else { return nil }

    // Look for the contiguous run that matches `phraseWords`.
    if phraseWords.count == 1 {
        guard let first = spans.first(where: { $0.normalizedWord == phraseWords[0] })
        else { return nil }
        return LocatedSpan(
            startTokenIndex: first.startTokenIndex, tokenCount: first.tokenCount)
    }
    let n = spans.count
    let k = phraseWords.count
    for start in 0...(n - k) {
        var match = true
        for offset in 0..<k {
            if spans[start + offset].normalizedWord != phraseWords[offset] {
                match = false
                break
            }
        }
        if match {
            let firstSpan = spans[start]
            let lastSpan = spans[start + k - 1]
            let tokenCount =
                (lastSpan.startTokenIndex + lastSpan.tokenCount) - firstSpan.startTokenIndex
            return LocatedSpan(
                startTokenIndex: firstSpan.startTokenIndex, tokenCount: tokenCount)
        }
    }
    return nil
}

// MARK: - Text rebuild

/// Apply word-level replacements onto the original transcript by iterating
/// in stream order and substituting the first match for each replacement.
/// CTC rescorer already preserves capitalization on `replacementWord`.
private func applyReplacementsToText(
    originalText: String, replacements: [VocabularyRescorer.RescoringResult]
) -> String {
    var text = originalText
    for r in replacements where r.shouldReplace {
        guard let to = r.replacementWord, !to.isEmpty else { continue }
        let from = r.originalWord
        guard let range = text.range(of: from) else { continue }
        text.replaceSubrange(range, with: to)
    }
    return text
}

// MARK: - Audio padding / windowing

private func padIfNeeded(_ samples: [Float], target: Int) -> [Float] {
    guard samples.count < target else { return samples }
    var padded = samples
    padded.append(contentsOf: [Float](repeating: 0, count: target - samples.count))
    return padded
}

/// Pick a sample window of size ≤ `maxSamples` that fully contains the
/// phrase range. Centers the window when possible and clamps to the
/// signal endpoints.
private func pickWindow(
    phraseStart: Int, phraseEnd: Int, totalSamples: Int, maxSamples: Int
) -> (start: Int, end: Int) {
    if totalSamples <= maxSamples {
        return (0, totalSamples)
    }
    let phraseLen = max(1, phraseEnd - phraseStart)
    if phraseLen >= maxSamples {
        let start = max(0, min(phraseStart, totalSamples - maxSamples))
        return (start, start + maxSamples)
    }
    let mid = (phraseStart + phraseEnd) / 2
    var start = mid - maxSamples / 2
    if start < 0 { start = 0 }
    if start + maxSamples > totalSamples { start = totalSamples - maxSamples }
    return (start, start + maxSamples)
}

/// Produce a shifted copy of `tokenTimings` whose times are expressed
/// relative to `sampleOffset` (i.e. as if the encoder run started there).
private func shiftTimings(
    _ tokenTimings: [TokenTiming], sampleOffset: Int
) -> [TokenTiming] {
    let dt = Double(sampleOffset) / Double(ASRConstants.sampleRate)
    if dt == 0 { return tokenTimings }
    return tokenTimings.map { t in
        TokenTiming(
            token: t.token,
            tokenId: t.tokenId,
            startTime: max(0, t.startTime - dt),
            endTime: max(0, t.endTime - dt),
            confidence: t.confidence
        )
    }
}
