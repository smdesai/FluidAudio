import Foundation

/// A prefix trie for efficient token-level keyword phrase matching during TDT decoding.
///
/// Built from `CustomVocabularyTerm` entries that have `tokenIds` populated.
/// At each decoding step, the trie is queried to determine if any keyword phrase
/// is being matched by the current token sequence.
struct KeywordPrefixTrie: Sendable {

    /// A single node in the trie.
    private struct Node: Sendable {
        /// Children keyed by token ID.
        var children: [Int: Int] = [:]  // token ID → node index
        /// When non-nil, this node marks the end of a keyword phrase at the given term index.
        var terminalTermIndex: Int?
    }

    /// Flat array of trie nodes (index 0 is the root).
    private let nodes: [Node]
    /// The vocabulary terms used to build this trie (for lookup by index).
    let terms: [CustomVocabularyTerm]

    /// Build a prefix trie from vocabulary terms that have `tokenIds` populated.
    ///
    /// Terms without `tokenIds` or with empty `tokenIds` are skipped.
    init(terms: [CustomVocabularyTerm]) {
        self.terms = terms
        var nodes = [Node()]  // Root node at index 0

        for (termIndex, term) in terms.enumerated() {
            guard let tokenIds = term.tokenIds, !tokenIds.isEmpty else { continue }

            var current = 0  // Start at root
            for tokenId in tokenIds {
                if let childIndex = nodes[current].children[tokenId] {
                    current = childIndex
                } else {
                    let newIndex = nodes.count
                    nodes.append(Node())
                    nodes[current].children[tokenId] = newIndex
                    current = newIndex
                }
            }
            // Mark terminal
            nodes[current].terminalTermIndex = termIndex
        }

        self.nodes = nodes
    }

    /// Returns the set of valid next token IDs that continue any keyword from the given prefix.
    func getValidNextTokens(prefix: [Int]) -> Set<Int> {
        var current = 0
        for tokenId in prefix {
            guard let childIndex = nodes[current].children[tokenId] else {
                return []
            }
            current = childIndex
        }
        return Set(nodes[current].children.keys)
    }

    /// Check if the given prefix completes a keyword phrase.
    func isComplete(prefix: [Int]) -> (matched: Bool, termIndex: Int?) {
        var current = 0
        for tokenId in prefix {
            guard let childIndex = nodes[current].children[tokenId] else {
                return (false, nil)
            }
            current = childIndex
        }
        if let termIndex = nodes[current].terminalTermIndex {
            return (true, termIndex)
        }
        return (false, nil)
    }

    /// Create a fresh cursor starting at the root.
    func makeCursor() -> TrieCursor {
        TrieCursor(nodeIndex: 0, prefix: [], trie: self)
    }

    /// Whether any terms were loaded into the trie.
    var isEmpty: Bool {
        nodes.count <= 1  // Only root node means no terms
    }

    // MARK: - Internal access for TrieCursor

    fileprivate func childIndex(of nodeIndex: Int, for tokenId: Int) -> Int? {
        guard nodeIndex < nodes.count else { return nil }
        return nodes[nodeIndex].children[tokenId]
    }

    fileprivate func terminalTermIndex(at nodeIndex: Int) -> Int? {
        guard nodeIndex < nodes.count else { return nil }
        return nodes[nodeIndex].terminalTermIndex
    }

    fileprivate func hasChildren(at nodeIndex: Int) -> Bool {
        guard nodeIndex < nodes.count else { return false }
        return !nodes[nodeIndex].children.isEmpty
    }

    fileprivate func childTokenIds(at nodeIndex: Int) -> Set<Int> {
        guard nodeIndex < nodes.count else { return [] }
        return Set(nodes[nodeIndex].children.keys)
    }
}

/// Lightweight cursor for tracking position in the trie without allocations per step.
struct TrieCursor: Sendable {
    /// Current node index in the trie.
    fileprivate let nodeIndex: Int
    /// Token IDs consumed so far on this path.
    let prefix: [Int]
    /// Reference to the owning trie.
    private let trie: KeywordPrefixTrie

    fileprivate init(nodeIndex: Int, prefix: [Int], trie: KeywordPrefixTrie) {
        self.nodeIndex = nodeIndex
        self.prefix = prefix
        self.trie = trie
    }

    /// Advance the cursor with a new token. Returns nil if no valid path exists.
    func advance(token: Int) -> TrieCursor? {
        guard let childIndex = trie.childIndex(of: nodeIndex, for: token) else {
            return nil
        }
        return TrieCursor(nodeIndex: childIndex, prefix: prefix + [token], trie: trie)
    }

    /// Whether this cursor is at a terminal node (a complete phrase).
    var isTerminal: Bool {
        trie.terminalTermIndex(at: nodeIndex) != nil
    }

    /// The matched term index if this cursor is at a terminal node.
    var matchedTermIndex: Int? {
        trie.terminalTermIndex(at: nodeIndex)
    }

    /// Valid next token IDs from this cursor position.
    var validNextTokens: Set<Int> {
        trie.childTokenIds(at: nodeIndex)
    }

    /// Whether there are any children from this position.
    var hasChildren: Bool {
        trie.hasChildren(at: nodeIndex)
    }
}
