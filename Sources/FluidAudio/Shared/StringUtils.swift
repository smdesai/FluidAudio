import Foundation

/// String and sequence utility functions
public enum StringUtils {
    /// Levenshtein (edit) distance between two sequences of equatable elements.
    /// Works for both character-level (String → [Character]) and word-level ([String]) comparisons.
    ///
    /// - Parameters:
    ///   - a: First sequence
    ///   - b: Second sequence
    /// - Returns: Minimum number of insertions, deletions, and substitutions to transform `a` into `b`
    public static func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        // Two rolling rows instead of a full (m+1)x(n+1) matrix. This is called
        // O(V * W) times per utterance by the vocabulary rescorer (once per
        // vocab term × TDT word), so the per-call 2D allocation dominated
        // large-vocab rescoring time. Rolling rows give identical results with
        // a single (n+1) buffer reused across rows.
        var previous = Array(0...n)
        var current = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            current[0] = i
            let ai = a[i - 1]
            for j in 1...n {
                let cost = ai == b[j - 1] ? 0 : 1
                current[j] = min(
                    previous[j] + 1,  // deletion
                    current[j - 1] + 1,  // insertion
                    previous[j - 1] + cost  // substitution
                )
            }
            swap(&previous, &current)
        }

        return previous[n]
    }

    /// Convenience overload for String comparison (character-level distance)
    public static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        return levenshteinDistance(Array(a), Array(b))
    }
}
