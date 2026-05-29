import Accelerate
@preconcurrency import CoreML
import Foundation

/// Beam-search variant of the TDT decoder.
///
/// Mirrors the structure of `TdtDecoderV3` (per-step joint scoring + per-
/// hypothesis duration-driven frame advancement), but maintains B parallel
/// hypotheses and selects from each step's top-K expansions. Replaces
/// `JointDecision` (which returns a pre-collapsed argmax) with
/// `JointDecisionLogits` (full per-step distributions) so the beam can
/// inspect alternative tokens.
///
/// Behaves as greedy decoding when `beamSize == 1`. The same model and
/// state types are used in both modes — the only difference is which
/// joint variant is loaded.
public struct TdtBeamDecoder: Sendable {

    private let logger = AppLogger(category: "TDT-beam")
    private let config: ASRConfig
    private let beamConfig: TdtBeamConfig
    private let modelInference = TdtModelInference()

    public init(config: ASRConfig, beamConfig: TdtBeamConfig) {
        self.config = config
        self.beamConfig = beamConfig
    }

    /// Public decode result. Mirrors the fields callers actually need from
    /// the internal `TdtHypothesis` while keeping the hypothesis type
    /// internal to the library.
    public struct DecodeResult: Sendable {
        public let tokens: [Int]
        public let timestamps: [Int]
        public let tokenConfidences: [Float]
        public let tokenDurations: [Int]
        public let score: Float
    }

    /// Public-API entry point. Mirrors `decode(...)` but returns a public
    /// `DecodeResult` instead of the internal `TdtHypothesis`.
    public func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointLogitsModel: MLModel,
        initialState: TdtDecoderState,
        globalFrameOffset: Int = 0,
        isLastChunk: Bool = false
    ) async throws -> DecodeResult {
        let hyp: TdtHypothesis = try await decode(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            decoderModel: decoderModel,
            jointLogitsModel: jointLogitsModel,
            initialState: initialState,
            globalFrameOffset: globalFrameOffset,
            isLastChunk: isLastChunk
        )
        return DecodeResult(
            tokens: hyp.ySequence,
            timestamps: hyp.timestamps,
            tokenConfidences: hyp.tokenConfidences,
            tokenDurations: hyp.tokenDurations,
            score: hyp.score
        )
    }

    /// Decode an utterance with beam search and return the best hypothesis.
    ///
    /// - Parameters:
    ///   - encoderOutput: Encoder activations [1, hiddenSize, T] in fp32.
    ///   - encoderSequenceLength: Valid frames in `encoderOutput`.
    ///   - decoderModel: TDT decoder LSTM (`Decoder.mlmodelc`).
    ///   - jointLogitsModel: `JointDecisionLogits.mlmodelc`. Required —
    ///     `JointDecision` (argmaxed) cannot drive a beam.
    ///   - initialState: LSTM state to seed the beam from. The caller still
    ///     gets back a `TdtHypothesis` with the winning beam's final state.
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointLogitsModel: MLModel,
        initialState: TdtDecoderState,
        globalFrameOffset: Int = 0,
        isLastChunk: Bool = false
    ) async throws -> TdtHypothesis {
        guard encoderSequenceLength > 1 else {
            return TdtHypothesis(decState: initialState)
        }

        let encoderHidden = config.encoderHiddenSize
        let encoderFrames = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderSequenceLength,
            expectedHiddenSize: encoderHidden
        )

        // Reusable joint inputs. Each beam expansion writes into these
        // before calling `runJointLogits`; the decoder then runs and we
        // copy out the projection into the hypothesis's owned buffer.
        let reusableEncoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: encoderHidden), 1], dataType: .float32
        )
        let reusableDecoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1], dataType: .float32
        )
        let encDestStride = reusableEncoderStep.strides.map { $0.intValue }[1]
        let encDestPtr = reusableEncoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: encoderHidden)

        // Reusable backings for the joint-logits outputs. CoreML writes
        // into these on each call instead of allocating fresh tensors,
        // which in turn lets `runJointLogits` skip the per-call output
        // allocation. Vocab-with-blank for TDT v2 is 1025; duration is 5.
        let vocabSize = config.tdtConfig.blankId + 1
        let durationBins = config.tdtConfig.durationBins.count
        let tokenLogitsBacking = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: vocabSize)] as [NSNumber], dataType: .float32)
        let durationLogitsBacking = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: durationBins)] as [NSNumber], dataType: .float32)

        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = 1

        let blankId = config.tdtConfig.blankId

        // Seed the beam with one SOS-primed hypothesis. SOS == blankId by
        // NeMo convention.
        var seedState = try TdtDecoderState(from: initialState)
        let seedDec = try modelInference.runDecoder(
            token: blankId,
            state: seedState,
            model: decoderModel,
            targetArray: reusableTargetArray,
            targetLengthArray: reusableTargetLengthArray
        )
        seedState = seedDec.newState
        let seedProjection = try extractDecoderProjection(from: seedDec.output)

        var beam: [TdtBeamHypothesis] = [
            TdtBeamHypothesis(
                tokens: [],
                timestamps: [],
                tokenConfidences: [],
                tokenDurations: [],
                logProb: 0,
                lastToken: initialState.lastToken,
                timeIndex: 0,
                state: seedState,
                lastDecoderProjection: seedProjection,
                biasMatches: [],
                consumedBiasWindows: [],
                symbolsAtCurrentFrame: 0
            )
        ]

        // Beam-search outer loop: each iteration extends every active
        // hypothesis by one emission step (token + duration). Hypotheses
        // that have consumed all encoder frames stop advancing.
        let maxIterations = encoderSequenceLength * beamConfig.maxSymbolsPerStep + 16
        var iteration = 0
        while iteration < maxIterations {
            iteration += 1

            // Partition: active = still has frames; finished = done.
            var active: [TdtBeamHypothesis] = []
            var finished: [TdtBeamHypothesis] = []
            for hyp in beam {
                if hyp.timeIndex >= encoderSequenceLength {
                    finished.append(hyp)
                } else {
                    active.append(hyp)
                }
            }
            if active.isEmpty {
                beam = finished
                break
            }

            // Expand each active hypothesis by its top-K next tokens.
            var expanded: [TdtBeamHypothesis] = []
            expanded.reserveCapacity(active.count * beamConfig.topKPerHypothesis)
            for hyp in active {
                try Task.checkCancellation()
                let extensions = try await expand(
                    hypothesis: hyp,
                    encoderFrames: encoderFrames,
                    decoderModel: decoderModel,
                    jointLogitsModel: jointLogitsModel,
                    encoderStep: reusableEncoderStep,
                    decoderStep: reusableDecoderStep,
                    encoderDestPtr: encDestPtr,
                    encoderDestStride: encDestStride,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray,
                    tokenLogitsBacking: tokenLogitsBacking,
                    durationLogitsBacking: durationLogitsBacking,
                    blankId: blankId
                )
                expanded.append(contentsOf: extensions)
            }

            // Re-merge with `finished` so they remain candidates for the
            // best-so-far comparison (a finished hypothesis with a high
            // log-prob can still beat ongoing ones).
            expanded.append(contentsOf: finished)

            // Prune by absolute log-prob and keep top-B.
            beam = prune(expanded)

            // Termination: if every kept hypothesis is finished, we're done.
            if beam.allSatisfy({ $0.timeIndex >= encoderSequenceLength }) {
                break
            }
        }

        // Pick the best hypothesis using length-normalized score.
        guard
            let best = beam.max(by: {
                $0.normalizedScore(lengthPenalty: beamConfig.lengthPenalty)
                    < $1.normalizedScore(lengthPenalty: beamConfig.lengthPenalty)
            })
        else {
            return TdtHypothesis(decState: initialState)
        }

        // Apply globalFrameOffset to timestamps for chunked decoding.
        var hyp = best.asTdtHypothesis()
        if globalFrameOffset != 0 {
            hyp.timestamps = hyp.timestamps.map { $0 + globalFrameOffset }
        }
        return hyp
    }

    // MARK: - Expansion

    /// Score one hypothesis at its current time index, generate up to
    /// `topKPerHypothesis` extensions, and return the new beams.
    private func expand(
        hypothesis: TdtBeamHypothesis,
        encoderFrames: EncoderFrameView,
        decoderModel: MLModel,
        jointLogitsModel: MLModel,
        encoderStep: MLMultiArray,
        decoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray,
        tokenLogitsBacking: MLMultiArray,
        durationLogitsBacking: MLMultiArray,
        blankId: Int
    ) async throws -> [TdtBeamHypothesis] {
        guard hypothesis.timeIndex < encoderFrames.count else {
            return [hypothesis]
        }

        // Copy this hypothesis's encoder frame and decoder projection into
        // the reusable buffers. The decoder projection is owned by the
        // hypothesis (siblings have their own copies).
        try encoderFrames.copyFrame(
            at: hypothesis.timeIndex,
            into: encoderDestPtr,
            destinationStride: encoderDestStride
        )
        guard let decoderProjection = hypothesis.lastDecoderProjection else {
            // Defensive — every hypothesis should always have a projection
            // either from seeding or from the last expansion.
            return [hypothesis]
        }
        try modelInference.normalizeDecoderProjection(decoderProjection, into: decoderStep)

        // Score the joint at (encoder[t], decoder_step). Pass the
        // pre-allocated output backings so CoreML writes into them
        // directly instead of allocating fresh tensors per call.
        let (tokenLogits, durationLogits) = try modelInference.runJointLogits(
            encoderStep: encoderStep,
            decoderStep: decoderStep,
            model: jointLogitsModel,
            tokenLogitsBacking: tokenLogitsBacking,
            durationLogitsBacking: durationLogitsBacking
        )

        // Apply shallow-fusion bias before log-softmax.
        var biasedLogits = tokenLogits
        if let bias = beamConfig.bias {
            applyBias(
                logits: &biasedLogits,
                hypothesis: hypothesis,
                bias: bias
            )
        }

        let tokenLogProbs = logSoftmax(biasedLogits)
        let durationLogProbs = logSoftmax(durationLogits)

        // Most-likely duration drives this step's frame advance. Beam
        // doesn't explore duration choices in v1 — that would multiply the
        // expansion factor by the duration count (5) for marginal gain.
        let durationBin = argmax(durationLogProbs)
        let durationLogProb = durationLogProbs[durationBin]
        var duration = try TdtDurationMapping.mapDurationBin(
            durationBin, durationBins: config.tdtConfig.durationBins)

        // Same duration=0 sanity rules as the greedy decoder.
        let topToken = argmax(tokenLogProbs)
        let isBlank = (topToken == blankId)
        if isBlank && duration == 0 { duration = 1 }
        if !isBlank && duration == 0
            && hypothesis.symbolsAtCurrentFrame >= beamConfig.maxSymbolsPerStep
        {
            duration = 1
        }

        // Blank-dominance shortcut. TDT typically emits blank on >90% of
        // frames. When the model is overwhelmingly confident the next
        // token is blank, skip the top-K expansion: emit only blank,
        // skipping K-1 hypothesis copies, K-1 decoder LSTM calls, and
        // the per-copy state allocation. WER impact is negligible because
        // those K-1 alternatives have log-prob more than `margin` below
        // the blank choice and would be dominated by other beams' main
        // emissions inside the same step.
        let topK: [Int]
        if isBlank, beamConfig.blankShortcutMargin.isFinite {
            // Find runner-up log-prob (largest non-blank).
            var runnerUp: Float = -.infinity
            for (idx, lp) in tokenLogProbs.enumerated() where idx != blankId {
                if lp > runnerUp { runnerUp = lp }
            }
            let blankLp = tokenLogProbs[blankId]
            if blankLp - runnerUp >= beamConfig.blankShortcutMargin {
                topK = [blankId]
            } else {
                topK = topKIndices(tokenLogProbs, k: beamConfig.topKPerHypothesis)
            }
        } else {
            topK = topKIndices(tokenLogProbs, k: beamConfig.topKPerHypothesis)
        }

        // Generate one expansion per top-K candidate.
        var extensions: [TdtBeamHypothesis] = []
        extensions.reserveCapacity(topK.count)
        for token in topK {
            let tokenLogProb = tokenLogProbs[token]
            let tokenIsBlank = (token == blankId)

            // Build the new hypothesis.
            var copy = hypothesis
            copy.logProb += tokenLogProb + durationLogProb

            if tokenIsBlank {
                // Blank: advance time, keep the decoder projection; do NOT
                // run the LSTM (matches `TdtDecoderV3`'s blank optimization).
                copy.timeIndex += duration
                if copy.timeIndex == hypothesis.timeIndex {
                    // Force advance to avoid infinite loop on blank+dur=0.
                    copy.timeIndex += 1
                }
                copy.symbolsAtCurrentFrame = 0
            } else {
                // Non-blank: emit, advance, run decoder LSTM with the new
                // token, refresh the projection.
                copy.tokens.append(token)
                copy.timestamps.append(hypothesis.timeIndex)
                copy.tokenConfidences.append(expf(tokenLogProb))
                copy.tokenDurations.append(duration)
                copy.lastToken = token

                var newState = try TdtDecoderState(from: copy.state)
                let step = try modelInference.runDecoder(
                    token: token,
                    state: newState,
                    model: decoderModel,
                    targetArray: targetArray,
                    targetLengthArray: targetLengthArray
                )
                newState = step.newState
                copy.state = newState
                copy.lastDecoderProjection = try extractDecoderProjection(from: step.output)

                if duration == 0 {
                    copy.symbolsAtCurrentFrame += 1
                } else {
                    copy.symbolsAtCurrentFrame = 0
                }
                copy.timeIndex += duration

                // Update bias state with the just-emitted token.
                if let bias = beamConfig.bias {
                    advanceBiasMatches(
                        on: &copy, lastToken: token, emittedFrame: hypothesis.timeIndex, bias: bias)
                }
            }

            extensions.append(copy)
        }
        return extensions
    }

    // MARK: - Pruning

    /// Drop hypotheses below absolute threshold and clip to beamSize.
    private func prune(_ candidates: [TdtBeamHypothesis]) -> [TdtBeamHypothesis] {
        guard !candidates.isEmpty else { return candidates }
        let bestLogProb = candidates.map { $0.logProb }.max() ?? 0
        let cutoff = bestLogProb - beamConfig.pruningThreshold
        let filtered = candidates.filter { $0.logProb >= cutoff }
        if filtered.count <= beamConfig.beamSize { return filtered }

        // Sort by length-normalized score (the same metric used for final
        // selection) so we don't keep long, mediocre hypotheses at the
        // expense of strong short ones.
        let sorted = filtered.sorted {
            $0.normalizedScore(lengthPenalty: beamConfig.lengthPenalty)
                > $1.normalizedScore(lengthPenalty: beamConfig.lengthPenalty)
        }
        return Array(sorted.prefix(beamConfig.beamSize))
    }

    // MARK: - Bias state machine

    /// Add `bias.bonus` to the next-token logit for every active match,
    /// and seed new matches starting at this hypothesis's last token.
    private func applyBias(
        logits: inout [Float],
        hypothesis: TdtBeamHypothesis,
        bias: TdtBeamBiasConfig
    ) {
        var boostedTokens = Set<Int>()
        // Existing matches: bonus the next expected token.
        for match in hypothesis.biasMatches {
            let seq = bias.keywordTokenSequences[match.keywordIndex]
            guard match.position < seq.count else { continue }
            let target = seq[match.position]
            if target >= 0 && target < logits.count {
                boostedTokens.insert(target)
            }
        }
        if bias.windows.isEmpty {
            // Global fallback: seed all first tokens. This is useful for tiny
            // dictionaries but risky for broad keyword lists.
            for (idx, seq) in bias.keywordTokenSequences.enumerated() {
                if hypothesis.biasMatches.contains(where: { $0.keywordIndex == idx }) {
                    continue
                }
                let first = seq[0]
                if first >= 0 && first < logits.count {
                    boostedTokens.insert(first)
                }
            }
        } else {
            let frame = hypothesis.timeIndex
            for (windowIndex, window) in bias.windows.enumerated() {
                guard !hypothesis.consumedBiasWindows.contains(windowIndex) else { continue }
                guard frame >= window.startFrame, frame <= window.endFrame else { continue }
                let idx = window.keywordIndex
                guard idx >= 0, idx < bias.keywordTokenSequences.count else { continue }
                if hypothesis.biasMatches.contains(where: { $0.keywordIndex == idx }) {
                    continue
                }
                let first = bias.keywordTokenSequences[idx][0]
                if first >= 0 && first < logits.count {
                    boostedTokens.insert(first)
                }
            }
        }
        for token in boostedTokens {
            logits[token] += bias.bonus
        }
    }

    /// After a non-blank emission, advance any matches whose expected
    /// next-token equals `lastToken`, drop matches that diverged, and
    /// activate new matches whose first token equals `lastToken`.
    private func advanceBiasMatches(
        on hypothesis: inout TdtBeamHypothesis,
        lastToken: Int,
        emittedFrame: Int,
        bias: TdtBeamBiasConfig
    ) {
        var updated: [TdtBeamBiasMatch] = []
        for match in hypothesis.biasMatches {
            let seq = bias.keywordTokenSequences[match.keywordIndex]
            guard match.position < seq.count else { continue }
            if seq[match.position] == lastToken {
                let nextPos = match.position + 1
                if nextPos < seq.count {
                    updated.append(
                        TdtBeamBiasMatch(
                            keywordIndex: match.keywordIndex, position: nextPos))
                }
                // Else: keyword complete, drop the match.
            }
            // Else: hypothesis diverged from this keyword; drop.
        }
        // Seed new matches. When CTC windows exist, only seed the keyword
        // whose detection window fired. Without this guard, broad vocabularies
        // let common first tokens activate hundreds of unrelated continuations.
        if bias.windows.isEmpty {
            for (idx, seq) in bias.keywordTokenSequences.enumerated() {
                seedBiasMatch(keywordIndex: idx, tokenSequence: seq, lastToken: lastToken, into: &updated)
            }
        } else {
            for (windowIndex, window) in bias.windows.enumerated() {
                guard !hypothesis.consumedBiasWindows.contains(windowIndex) else { continue }
                guard emittedFrame >= window.startFrame, emittedFrame <= window.endFrame else { continue }
                let idx = window.keywordIndex
                guard idx >= 0, idx < bias.keywordTokenSequences.count else { continue }
                seedBiasMatch(
                    keywordIndex: idx,
                    tokenSequence: bias.keywordTokenSequences[idx],
                    lastToken: lastToken,
                    into: &updated
                )
            }
        }
        hypothesis.biasMatches = updated

        if !bias.windows.isEmpty {
            for (windowIndex, window) in bias.windows.enumerated() {
                guard !hypothesis.consumedBiasWindows.contains(windowIndex) else { continue }
                guard emittedFrame >= window.startFrame, emittedFrame <= window.endFrame else { continue }
                let idx = window.keywordIndex
                guard idx >= 0, idx < bias.keywordTokenSequences.count else { continue }
                if bias.keywordTokenSequences[idx][0] == lastToken {
                    hypothesis.consumedBiasWindows.insert(windowIndex)
                }
            }
        }
    }

    private func seedBiasMatch(
        keywordIndex: Int,
        tokenSequence: [Int],
        lastToken: Int,
        into matches: inout [TdtBeamBiasMatch]
    ) {
        guard tokenSequence[0] == lastToken else { return }
        guard tokenSequence.count > 1 else { return }
        guard !matches.contains(where: { $0.keywordIndex == keywordIndex && $0.position == 1 }) else { return }
        matches.append(TdtBeamBiasMatch(keywordIndex: keywordIndex, position: 1))
    }

    // MARK: - Helpers

    private func extractDecoderProjection(from output: MLFeatureProvider) throws -> MLMultiArray {
        guard let projection = output.featureValue(for: "decoder")?.multiArrayValue else {
            throw ASRError.processingFailed("Beam decoder: missing 'decoder' projection")
        }
        return projection
    }

    private func logSoftmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        var maxVal: Float = -.infinity
        for v in logits where v > maxVal { maxVal = v }
        var sumExp: Float = 0
        for v in logits { sumExp += expf(v - maxVal) }
        let logSumExp = logf(sumExp)
        return logits.map { ($0 - maxVal) - logSumExp }
    }

    private func argmax(_ values: [Float]) -> Int {
        var bestIdx = 0
        var bestVal = values.first ?? -.infinity
        for (i, v) in values.enumerated() where v > bestVal {
            bestVal = v
            bestIdx = i
        }
        return bestIdx
    }

    /// Return indices of the top-K largest values, sorted descending. Uses
    /// a bounded heap-by-min — fine for K up to ~16.
    private func topKIndices(_ values: [Float], k: Int) -> [Int] {
        let cap = min(k, values.count)
        guard cap > 0 else { return [] }
        var top: [(Int, Float)] = []
        top.reserveCapacity(cap + 1)
        for (i, v) in values.enumerated() {
            if top.count < cap {
                top.append((i, v))
                if top.count == cap {
                    top.sort { $0.1 < $1.1 }  // ascending — top[0] is the worst
                }
            } else if v > top[0].1 {
                top[0] = (i, v)
                // Re-bubble: only top[0] changed, restore ascending order.
                var j = 0
                while j + 1 < cap && top[j].1 > top[j + 1].1 {
                    top.swapAt(j, j + 1)
                    j += 1
                }
            }
        }
        // Final descending order.
        return top.sorted { $0.1 > $1.1 }.map { $0.0 }
    }
}
