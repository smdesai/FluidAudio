import Foundation

/// Shared-prefix CTC context graph for custom vocabulary spotting.
///
/// This implements the core shape described in arXiv:2406.07096: vocabulary
/// token sequences are compiled into a trie, and decoding runs over the trie
/// composed with CTC blank/token topology. Unlike scoring each phrase with an
/// independent DP table, shared prefixes compete in one graph search.
struct CtcContextGraph: Sendable {

    struct Entry {
        let term: CustomVocabularyTerm
        let tokenIds: [Int]
        let minScore: Float
    }

    struct Detection {
        let entry: Entry
        let score: Float
        let startFrame: Int
        let endFrame: Int
    }

    private struct Node {
        let token: Int?
        var children: [Int: Int]
        var entryIndices: [Int]
    }

    private enum Phase: Hashable {
        case blank
        case token
    }

    private struct StateKey: Hashable {
        let nodeIndex: Int
        let phase: Phase
    }

    private struct Hypothesis {
        let key: StateKey
        let score: Float
        let startFrame: Int
        let lastTokenFrame: Int
    }

    private let nodes: [Node]
    private let entries: [Entry]
    private let blankId: Int
    private let contextScore: Float

    init(
        vocabulary: CustomVocabularyContext,
        minScore: Float?,
        blankId: Int,
        contextScore: Float = 0
    ) {
        var nodes = [Node(token: nil, children: [:], entryIndices: [])]
        var entries: [Entry] = []

        for term in vocabulary.terms {
            guard term.text.count >= vocabulary.minTermLength else { continue }
            guard let tokenIds = term.ctcTokenIds ?? term.tokenIds, !tokenIds.isEmpty else { continue }
            guard !tokenIds.contains(ContextBiasingConstants.wildcardTokenId) else { continue }

            let adjustedMinScore = Self.adjustedThreshold(base: minScore, tokenCount: tokenIds.count)
            let entryIndex = entries.count
            entries.append(Entry(term: term, tokenIds: tokenIds, minScore: adjustedMinScore))

            var nodeIndex = 0
            for token in tokenIds {
                if let existing = nodes[nodeIndex].children[token] {
                    nodeIndex = existing
                    continue
                }

                let childIndex = nodes.count
                nodes.append(Node(token: token, children: [:], entryIndices: []))
                nodes[nodeIndex].children[token] = childIndex
                nodeIndex = childIndex
            }
            nodes[nodeIndex].entryIndices.append(entryIndex)
        }

        self.nodes = nodes
        self.entries = entries
        self.blankId = blankId
        self.contextScore = contextScore
    }

    var isEmpty: Bool { entries.isEmpty }

    func spot(logProbs: [[Float]], mergeOverlap: Bool = true) -> [Detection] {
        guard !entries.isEmpty, !logProbs.isEmpty else { return [] }

        var active: [StateKey: Hypothesis] = [:]
        var detections: [Detection] = []

        for (frameIndex, frame) in logProbs.enumerated() {
            var next: [StateKey: Hypothesis] = [:]

            let root = Hypothesis(
                key: StateKey(nodeIndex: 0, phase: .blank),
                score: 0,
                startFrame: frameIndex,
                lastTokenFrame: frameIndex
            )

            advance(hypothesis: root, frame: frame, frameIndex: frameIndex, into: &next)
            for hypothesis in active.values {
                advance(hypothesis: hypothesis, frame: frame, frameIndex: frameIndex, into: &next)
            }

            for hypothesis in next.values {
                appendTerminalDetections(for: hypothesis, into: &detections)
            }

            active = next
        }

        guard mergeOverlap else { return detections }
        return mergeOverlappingDetections(detections)
    }

    private func advance(
        hypothesis: Hypothesis,
        frame: [Float],
        frameIndex: Int,
        into next: inout [StateKey: Hypothesis]
    ) {
        let node = nodes[hypothesis.key.nodeIndex]

        if hypothesis.key.nodeIndex != 0, blankId >= 0, blankId < frame.count {
            insertBest(
                Hypothesis(
                    key: StateKey(nodeIndex: hypothesis.key.nodeIndex, phase: .blank),
                    score: hypothesis.score + frame[blankId],
                    startFrame: hypothesis.startFrame,
                    lastTokenFrame: hypothesis.lastTokenFrame
                ),
                into: &next
            )
        }

        if hypothesis.key.phase == .token, let token = node.token, token >= 0, token < frame.count {
            insertBest(
                Hypothesis(
                    key: StateKey(nodeIndex: hypothesis.key.nodeIndex, phase: .token),
                    score: hypothesis.score + frame[token],
                    startFrame: hypothesis.startFrame,
                    lastTokenFrame: frameIndex
                ),
                into: &next
            )
        }

        for (token, childIndex) in node.children where token >= 0 && token < frame.count {
            if hypothesis.key.phase == .token, node.token == token {
                continue
            }

            insertBest(
                Hypothesis(
                    key: StateKey(nodeIndex: childIndex, phase: .token),
                    score: hypothesis.score + frame[token] + contextScore,
                    startFrame: hypothesis.key.nodeIndex == 0 ? frameIndex : hypothesis.startFrame,
                    lastTokenFrame: frameIndex
                ),
                into: &next
            )
        }
    }

    private func insertBest(_ candidate: Hypothesis, into next: inout [StateKey: Hypothesis]) {
        if let existing = next[candidate.key], existing.score >= candidate.score {
            return
        }
        next[candidate.key] = candidate
    }

    private func appendTerminalDetections(for hypothesis: Hypothesis, into detections: inout [Detection]) {
        let node = nodes[hypothesis.key.nodeIndex]
        guard !node.entryIndices.isEmpty else { return }

        for entryIndex in node.entryIndices {
            let entry = entries[entryIndex]
            let normFactor = max(1, CtcDPAlgorithm.nonWildcardCount(entry.tokenIds))
            let normalizedScore = hypothesis.score / Float(normFactor)
            guard normalizedScore >= entry.minScore else { continue }

            detections.append(
                Detection(
                    entry: entry,
                    score: normalizedScore,
                    startFrame: hypothesis.startFrame,
                    endFrame: hypothesis.lastTokenFrame
                ))
        }
    }

    private func mergeOverlappingDetections(_ detections: [Detection]) -> [Detection] {
        let grouped = Dictionary(grouping: detections) { $0.entry.term.textLowercased }
        var merged: [Detection] = []
        for group in grouped.values {
            let sorted = group.sorted { lhs, rhs in
                if lhs.startFrame != rhs.startFrame { return lhs.startFrame < rhs.startFrame }
                return lhs.score > rhs.score
            }
            var termMerged: [Detection] = []
            for detection in sorted {
                if let last = termMerged.last, detection.startFrame <= last.endFrame {
                    var best = detection.score > last.score ? detection : last
                    best = Detection(
                        entry: best.entry,
                        score: best.score,
                        startFrame: min(last.startFrame, detection.startFrame),
                        endFrame: max(last.endFrame, detection.endFrame)
                    )
                    termMerged[termMerged.count - 1] = best
                    continue
                }
                termMerged.append(detection)
            }
            merged.append(contentsOf: termMerged)
        }
        return merged.sorted { lhs, rhs in
            if lhs.startFrame != rhs.startFrame { return lhs.startFrame < rhs.startFrame }
            return lhs.score > rhs.score
        }
    }

    private static func adjustedThreshold(base: Float?, tokenCount: Int) -> Float {
        let baseThreshold = base ?? ContextBiasingConstants.defaultMinSpotterScore
        let extraTokens = max(0, tokenCount - ContextBiasingConstants.baselineTokenCountForThreshold)
        return baseThreshold - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
    }
}
