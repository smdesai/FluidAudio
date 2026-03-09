import Foundation

/// Tokenizer built from the Parakeet TDT vocabulary dictionary that enumerates
/// all valid BPE decompositions of a phrase.
///
/// The TDT vocab uses space-prefixed BPE subwords (e.g. `" pro"`, `"er"`, `" the"`).
/// Since no `tokenizer.json` or SentencePiece model exists for this model, we cannot
/// know which specific BPE merge order the decoder will use. Instead, we enumerate
/// ALL valid token-ID paths that reconstruct the phrase and insert each path into the
/// trie so whichever decomposition the RNNT decoder happens to emit will match.
public struct TdtGreedyTokenizer: Sendable {
    /// Reverse lookup from token text to token ID.
    private let tokenToId: [String: Int]
    /// Maximum character length of any token in the vocabulary.
    private let maxTokenLength: Int

    /// Build the tokenizer from the ASR model's vocabulary dictionary.
    ///
    /// - Parameter vocabulary: The `[Int: String]` vocabulary from `AsrModels.vocabulary`
    public init(vocabulary: [Int: String]) {
        var lookup: [String: Int] = [:]
        var maxLen = 0

        for (id, token) in vocabulary {
            // Skip special tokens (e.g. "<blank>", "<eos>")
            guard !token.hasPrefix("<") else { continue }
            lookup[token] = id
            maxLen = max(maxLen, token.count)
        }

        self.tokenToId = lookup
        self.maxTokenLength = maxLen
    }

    /// Encode a phrase into TDT token IDs using greedy longest-match.
    ///
    /// Returns a single tokenization path (the greediest). For keyword boosting
    /// use `encodeAllPaths(_:)` instead which covers all valid decompositions.
    ///
    /// - Parameter text: The phrase to tokenize (e.g. "cancel order")
    /// - Returns: Array of token IDs, or empty if the text cannot be tokenized
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty, maxTokenLength > 0 else { return [] }

        let input = " " + text.lowercased()
        let chars = Array(input)
        var tokenIds: [Int] = []
        var pos = 0

        while pos < chars.count {
            var matched = false
            let remaining = chars.count - pos
            let maxLen = min(maxTokenLength, remaining)

            for length in stride(from: maxLen, through: 1, by: -1) {
                let candidate = String(chars[pos..<(pos + length)])
                if let id = tokenToId[candidate] {
                    tokenIds.append(id)
                    pos += length
                    matched = true
                    break
                }
            }

            if !matched {
                pos += 1
            }
        }

        return tokenIds
    }

    /// Enumerate all valid tokenizations of a phrase using breadth-first search.
    ///
    /// Because the RNNT decoder can emit ANY valid BPE decomposition (not necessarily
    /// the same one as greedy longest-match), we enumerate all paths so the trie
    /// matches regardless of which decomposition the decoder uses.
    ///
    /// BFS is used so that the shortest paths (fewest tokens = most merged subwords)
    /// are found first. This gives the best coverage within the path limit because
    /// BPE tokenizers tend to produce highly-merged decompositions.
    ///
    /// - Parameters:
    ///   - text: The phrase to tokenize (e.g. "cancel order")
    ///   - maxPaths: Maximum number of paths to return (default: 64)
    /// - Returns: Array of token-ID sequences, each representing a valid decomposition
    public func encodeAllPaths(_ text: String, maxPaths: Int = 64) -> [[Int]] {
        guard !text.isEmpty, maxTokenLength > 0 else { return [] }

        let input = " " + text.lowercased()
        let chars = Array(input)
        let n = chars.count

        // For each position, store all tokens that start there
        var tokensAt: [[TokenMatch]] = Array(repeating: [], count: n)

        for pos in 0..<n {
            let remaining = n - pos
            let maxLen = min(maxTokenLength, remaining)
            for length in 1...maxLen {
                let candidate = String(chars[pos..<(pos + length)])
                if let id = tokenToId[candidate] {
                    tokensAt[pos].append(TokenMatch(id: id, length: length))
                }
            }
        }

        // BFS from position 0 to collect paths reaching position n.
        // Shortest paths (fewest tokens) come first — these are the most merged
        // decompositions and most likely to match BPE output.
        var results: [[Int]] = []
        var queue: [(pos: Int, path: [Int])] = [(0, [])]
        var head = 0

        while head < queue.count {
            guard results.count < maxPaths else { break }
            let (pos, path) = queue[head]
            head += 1

            if pos == n {
                results.append(path)
                continue
            }

            for match in tokensAt[pos] {
                queue.append((pos + match.length, path + [match.id]))
            }
        }

        return results
    }
}

/// A token that matches at a given position in the input string.
private struct TokenMatch {
    let id: Int
    let length: Int
}
