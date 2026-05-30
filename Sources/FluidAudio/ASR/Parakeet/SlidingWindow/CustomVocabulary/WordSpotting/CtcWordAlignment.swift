import Foundation

/// Greedy CTC word-level alignment with per-word acoustic scores.
///
/// CTC-WS uses this alignment as the competing baseline for context-graph
/// candidates. Each aligned word records the collapsed token IDs, frame range,
/// and a score formed from greedy non-blank token log-probs plus the CTC
/// alignment token weight described in arXiv:2406.07096.
public struct CtcWordAlignment: Sendable {
    public let word: String
    public let tokenIds: [Int]
    public let score: Float
    public let startFrame: Int
    public let endFrame: Int
    public let startTime: TimeInterval
    public let endTime: TimeInterval

    public var normalizedScore: Float {
        guard !tokenIds.isEmpty else { return score }
        return score / Float(tokenIds.count)
    }

    public init(
        word: String,
        tokenIds: [Int],
        score: Float,
        startFrame: Int,
        endFrame: Int,
        startTime: TimeInterval,
        endTime: TimeInterval
    ) {
        self.word = word
        self.tokenIds = tokenIds
        self.score = score
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.startTime = startTime
        self.endTime = endTime
    }
}

enum CtcWordAligner {

    private struct TokenEvent {
        let tokenId: Int
        let piece: String
        let score: Float
        let frame: Int
    }

    static func align(
        logProbs: [[Float]],
        vocabulary: [Int: String],
        blankId: Int,
        frameDuration: Double,
        tokenWeight: Float = ContextBiasingConstants.defaultCtcAlignmentTokenWeight
    ) -> [CtcWordAlignment] {
        let events = collapsedTokenEvents(
            logProbs: logProbs,
            vocabulary: vocabulary,
            blankId: blankId,
            tokenWeight: tokenWeight
        )
        return buildWordAlignments(from: events, frameDuration: frameDuration)
    }

    private static func collapsedTokenEvents(
        logProbs: [[Float]],
        vocabulary: [Int: String],
        blankId: Int,
        tokenWeight: Float
    ) -> [TokenEvent] {
        var events: [TokenEvent] = []
        var previousBest = -1

        for (frameIndex, frame) in logProbs.enumerated() {
            guard let (bestId, bestScore) = argmax(frame) else { continue }
            defer { previousBest = bestId }
            guard bestId != blankId, bestId != previousBest else { continue }
            guard let piece = vocabulary[bestId], !piece.isEmpty else { continue }

            events.append(
                TokenEvent(
                    tokenId: bestId,
                    piece: piece,
                    score: bestScore + tokenWeight,
                    frame: frameIndex
                ))
        }

        return events
    }

    private static func buildWordAlignments(
        from events: [TokenEvent],
        frameDuration: Double
    ) -> [CtcWordAlignment] {
        var alignments: [CtcWordAlignment] = []
        var pieces: [String] = []
        var tokenIds: [Int] = []
        var score: Float = 0
        var startFrame: Int?
        var endFrame: Int?

        func flush() {
            guard let firstFrame = startFrame, let lastFrame = endFrame else { return }
            let word = pieces.joined().trimmingCharacters(in: .whitespacesAndNewlines)
            guard !word.isEmpty else {
                pieces.removeAll(keepingCapacity: true)
                tokenIds.removeAll(keepingCapacity: true)
                score = 0
                startFrame = nil
                endFrame = nil
                return
            }

            alignments.append(
                CtcWordAlignment(
                    word: word,
                    tokenIds: tokenIds,
                    score: score,
                    startFrame: firstFrame,
                    endFrame: lastFrame,
                    startTime: TimeInterval(firstFrame) * frameDuration,
                    endTime: TimeInterval(lastFrame) * frameDuration
                ))

            pieces.removeAll(keepingCapacity: true)
            tokenIds.removeAll(keepingCapacity: true)
            score = 0
            startFrame = nil
            endFrame = nil
        }

        for event in events {
            let startsNewWord =
                event.piece.hasPrefix(ASRConstants.sentencePieceWordBoundary) || event.piece.hasPrefix(" ")
            if startsNewWord {
                flush()
            }

            let piece = stripWordBoundaryPrefix(event.piece)
            if startFrame == nil { startFrame = event.frame }
            endFrame = event.frame
            pieces.append(piece)
            tokenIds.append(event.tokenId)
            score += event.score
        }
        flush()

        return alignments
    }

    private static func argmax(_ values: [Float]) -> (Int, Float)? {
        guard var bestValue = values.first else { return nil }
        var bestIndex = 0
        for index in values.indices.dropFirst() where values[index] > bestValue {
            bestValue = values[index]
            bestIndex = index
        }
        return (bestIndex, bestValue)
    }
}

enum CtcAlignmentValidator {

    static func candidateBeatsGreedyAlignment(
        candidateScore: Float,
        candidateStartFrame: Int,
        candidateEndFrame: Int,
        alignments: [CtcWordAlignment]
    ) -> Bool {
        guard !alignments.isEmpty else { return true }

        let overlapping = alignments.filter { alignment in
            alignment.endFrame >= candidateStartFrame && alignment.startFrame <= candidateEndFrame
        }
        guard !overlapping.isEmpty else { return true }

        let greedyScore = overlapping.reduce(-Float.infinity) { partial, alignment in
            max(partial, alignment.normalizedScore)
        }
        return candidateScore > greedyScore
    }

    /// Large-vocabulary false-accept veto for the primary term-centric path.
    ///
    /// The size-gated spotter-rescue pass already validates its detections
    /// against the greedy CTC word alignment, but the primary string-similarity
    /// path does not — so for large keyword lists a `+cbw`-boosted distractor
    /// (e.g. `prior` → `priorix`) can replace a correctly-decoded common word
    /// with no acoustic check. This applies the same veto on that path, but
    /// only above `largeVocabThreshold`: the small-dictionary path is already at
    /// 100% precision and is intentionally left untouched.
    ///
    /// - Parameters:
    ///   - boostedVocabScore: per-token vocab CTC score with cbw already added
    ///     (same scale as `CtcWordAlignment.normalizedScore`).
    ///   - candidateStartFrame/candidateEndFrame: the matched CTC frame interval.
    ///   - alignments: greedy CTC word alignment for the utterance.
    ///   - vocabularyTermCount: number of terms in the active vocabulary.
    ///   - largeVocabThreshold: the size above which the veto activates.
    /// - Returns: `true` if the replacement may proceed; `false` if vetoed.
    static func candidatePassesLargeVocabAlignmentVeto(
        boostedVocabScore: Float,
        candidateStartFrame: Int,
        candidateEndFrame: Int,
        alignments: [CtcWordAlignment],
        vocabularyTermCount: Int,
        largeVocabThreshold: Int
    ) -> Bool {
        guard vocabularyTermCount > largeVocabThreshold else { return true }
        return candidateBeatsGreedyAlignment(
            candidateScore: boostedVocabScore,
            candidateStartFrame: candidateStartFrame,
            candidateEndFrame: candidateEndFrame,
            alignments: alignments
        )
    }

    static func bestOverlappingGreedyScore(
        candidateStartFrame: Int,
        candidateEndFrame: Int,
        alignments: [CtcWordAlignment]
    ) -> Float? {
        let overlapping = alignments.filter { alignment in
            alignment.endFrame >= candidateStartFrame && alignment.startFrame <= candidateEndFrame
        }
        guard !overlapping.isEmpty else { return nil }
        return overlapping.reduce(-Float.infinity) { partial, alignment in
            max(partial, alignment.normalizedScore)
        }
    }
}
