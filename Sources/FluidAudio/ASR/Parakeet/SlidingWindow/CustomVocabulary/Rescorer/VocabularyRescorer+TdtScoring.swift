@preconcurrency import CoreML
import Foundation

// MARK: - TDT Scoring Context

extension VocabularyRescorer {

    /// One pre-computed encoder run for a sample window. The rescorer
    /// will pick the entry whose window contains the candidate's phrase
    /// time range and shift token timings into the run's local frame
    /// coordinates before scoring.
    public struct TdtEncoderRun: Sendable {
        public let encoder: MLMultiArray
        public let validLength: Int
        public let sampleStart: Int
        public let sampleEnd: Int

        public init(
            encoder: MLMultiArray,
            validLength: Int,
            sampleStart: Int,
            sampleEnd: Int
        ) {
            self.encoder = encoder
            self.validLength = validLength
            self.sampleStart = sampleStart
            self.sampleEnd = sampleEnd
        }
    }

    /// Context required to perform TDT-side acoustic scoring inside the
    /// rescorer. The caller pre-computes one or more `TdtEncoderRun`
    /// entries (typically one per word window) and the rescorer picks the
    /// one that contains each candidate's phrase. The lookup is sync so
    /// the rescorer's evaluator stays sync.
    public struct TdtScorerContext {

        /// The TDT-side rescorer that owns the joint-logits model and the
        /// SentencePiece tokenizer.
        let scorer: TdtRescorer

        /// All TDT token timings for the utterance — needed because the
        /// scorer replays the decoder LSTM over the prefix tokens and
        /// reads candidate-original tokens by absolute index.
        let tokenTimings: [TokenTiming]

        /// Pre-computed encoder runs covering the audio. The rescorer
        /// picks an entry whose `[sampleStart, sampleEnd)` range contains
        /// the candidate phrase. Caller should ensure the runs jointly
        /// cover the full transcript time range.
        let encoderRuns: [TdtEncoderRun]

        /// Sample rate of the audio used to build encoderRuns. Used to
        /// convert phrase times into sample indices for window lookup.
        let sampleRate: Int

        /// Margin (log-prob units) by which the candidate must outscore
        /// the original to flip a replacement. Mirrors the CTC scorer's
        /// "vocab_score > original_score" decision but with a slack so
        /// borderline TDT decisions don't flip-flop.
        let acceptMargin: Float

        /// Minimum confidence in the candidate sequence before a positive
        /// decision is trusted. When TDT is very uncertain in *both*
        /// hypotheses the comparison is unreliable; bail out and keep the
        /// original.
        let minCandidateScore: Float

        public init(
            scorer: TdtRescorer,
            tokenTimings: [TokenTiming],
            encoderRuns: [TdtEncoderRun],
            sampleRate: Int = ASRConstants.sampleRate,
            acceptMargin: Float = 0.0,
            minCandidateScore: Float = -.infinity
        ) {
            self.scorer = scorer
            self.tokenTimings = tokenTimings
            self.encoderRuns = encoderRuns
            self.sampleRate = sampleRate
            self.acceptMargin = acceptMargin
            self.minCandidateScore = minCandidateScore
        }

        /// Pick the encoder run whose window contains the phrase. Prefers
        /// the run with the most context to either side of the phrase.
        /// Returns `nil` if no run contains the phrase.
        func runContaining(phraseStart: Double, phraseEnd: Double) -> TdtEncoderRun? {
            let phraseStartSample = Int(phraseStart * Double(sampleRate))
            let phraseEndSample = Int(phraseEnd * Double(sampleRate))
            // Choose the run that maximizes how centered the phrase is.
            var best: (run: TdtEncoderRun, slack: Int)? = nil
            for run in encoderRuns {
                guard run.sampleStart <= phraseStartSample,
                    run.sampleEnd >= phraseEndSample
                else { continue }
                let leftSlack = phraseStartSample - run.sampleStart
                let rightSlack = run.sampleEnd - phraseEndSample
                let slack = min(leftSlack, rightSlack)
                if best == nil || slack > best!.slack {
                    best = (run, slack)
                }
            }
            return best?.run
        }
    }

    // MARK: - TDT Match Evaluation

    /// Score a candidate replacement against the original phrase using the
    /// TDT posterior. Returns `nil` when scoring fails (e.g. tokenizer
    /// produces no tokens) so the caller can fall back to the CTC path.
    func evaluateTDTMatch(
        candidate: CTCMatchCandidate,
        wordTimings: [WordTiming],
        context: TdtScorerContext
    ) -> CTCMatchResult? {
        guard
            let firstWordIdx = candidate.spanIndices.first,
            let lastWordIdx = candidate.spanIndices.last,
            firstWordIdx < wordTimings.count,
            lastWordIdx < wordTimings.count
        else { return nil }

        let firstWord = wordTimings[firstWordIdx]
        let lastWord = wordTimings[lastWordIdx]
        let phraseStartToken = firstWord.tokenStartIndex
        let phraseTokenCount =
            (lastWord.tokenStartIndex + lastWord.tokenCount) - phraseStartToken
        guard phraseTokenCount > 0,
            phraseStartToken + phraseTokenCount <= context.tokenTimings.count
        else { return nil }

        // Locate the encoder run whose window contains this phrase, then
        // shift token timings into that run's local frame coordinates.
        guard
            let run = context.runContaining(
                phraseStart: candidate.spanStartTime, phraseEnd: candidate.spanEndTime)
        else {
            debugLog(
                "  [TDT] no encoder run covers phrase '\(candidate.originalPhrase)' "
                    + "[\(String(format: "%.2f", candidate.spanStartTime))-\(String(format: "%.2f", candidate.spanEndTime))s]; "
                    + "falling back to CTC"
            )
            return nil
        }
        let runOffset = Double(run.sampleStart) / Double(context.sampleRate)
        let shiftedTimings: [TokenTiming]
        if runOffset == 0 {
            shiftedTimings = context.tokenTimings
        } else {
            shiftedTimings = context.tokenTimings.map { t in
                TokenTiming(
                    token: t.token,
                    tokenId: t.tokenId,
                    startTime: max(0, t.startTime - runOffset),
                    endTime: max(0, t.endTime - runOffset),
                    confidence: t.confidence
                )
            }
        }

        let decision: TdtRescorer.Decision
        do {
            decision = try context.scorer.score(
                originalPhrase: candidate.originalPhrase,
                candidate: candidate.vocabTerm,
                tokenTimings: shiftedTimings,
                phraseStartTokenIndex: phraseStartToken,
                phraseTokenCount: phraseTokenCount,
                encoderOutput: run.encoder,
                encoderSequenceLength: run.validLength
            )
        } catch {
            debugLog("  [TDT] score failed for '\(candidate.originalPhrase)' → '\(candidate.vocabTerm)': \(error)")
            return nil
        }

        // Accept-side gate: candidate must strictly beat original by
        // `acceptMargin` log-prob points AND meet the minimum-confidence
        // floor. Without these, every CTC-similarity-matched candidate
        // whose score is even infinitesimally above the original would
        // flip; in practice the TDT logits are noisy at the 1-2 point
        // level and we don't want flip-flops.
        let margin = decision.candidateScore - decision.originalScore
        let marginPasses = margin >= context.acceptMargin
        let confidencePasses = decision.candidateScore >= context.minCandidateScore
        let shouldReplace = marginPasses && confidencePasses

        let firstOriginalWord =
            candidate.originalPhrase.split(separator: " ").first.map(String.init)
            ?? candidate.originalPhrase
        let replacement = preserveCapitalization(
            original: firstOriginalWord, replacement: candidate.vocabTerm)

        let outcome: String
        if shouldReplace {
            outcome = "REPLACE"
        } else if !marginPasses {
            outcome = "below margin"
        } else {
            outcome = "below confidence"
        }
        debugLog(
            "  [TDT] '\(candidate.originalPhrase)' vs '\(candidate.vocabTerm)' "
                + "orig=\(String(format: "%.2f", decision.originalScore)) "
                + "cand=\(String(format: "%.2f", decision.candidateScore)) "
                + "margin=\(String(format: "%.2f", margin)) → \(outcome)"
        )

        let reason =
            "TDT-vs-TDT: '\(candidate.vocabTerm)'=\(String(format: "%.2f", decision.candidateScore)) "
            + "vs '\(candidate.originalPhrase)'=\(String(format: "%.2f", decision.originalScore)) "
            + "margin=\(String(format: "%.2f", margin))"

        return CTCMatchResult(
            shouldReplace: shouldReplace,
            originalScore: decision.originalScore,
            boostedVocabScore: decision.candidateScore,
            replacement: replacement,
            reason: reason
        )
    }
}

// MARK: - Dispatch wrapper

extension VocabularyRescorer {

    /// Choose between the CTC and TDT match evaluators. When `tdtContext`
    /// is supplied and TDT scoring succeeds, its result is used; otherwise
    /// we fall back to the existing CTC scoring path. Falling back rather
    /// than failing keeps individual TDT errors from killing the whole
    /// rescore.
    func evaluateMatch(
        candidate: CTCMatchCandidate,
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        tdtContext: TdtScorerContext?,
        ctcWordAlignments: [CtcWordAlignment] = []
    ) -> CTCMatchResult {
        if let tdtContext,
            let tdtResult = evaluateTDTMatch(
                candidate: candidate,
                wordTimings: wordTimings,
                context: tdtContext
            )
        {
            return tdtResult
        }
        return evaluateCTCMatch(
            candidate: candidate,
            logProbs: logProbs,
            frameDuration: frameDuration,
            cbw: cbw,
            marginSeconds: marginSeconds,
            ctcWordAlignments: ctcWordAlignments
        )
    }
}

// Tiny helper that mirrors the private one inside +TokenEvaluation.swift,
// kept here so this extension compiles standalone. It only touches the
// member already exposed by `VocabularyRescorer`.
extension VocabularyRescorer {
    @inline(__always)
    fileprivate func debugLog(_ message: @escaping @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }
}
