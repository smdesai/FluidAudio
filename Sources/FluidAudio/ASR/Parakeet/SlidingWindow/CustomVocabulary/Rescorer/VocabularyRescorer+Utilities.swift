import Foundation

extension VocabularyRescorer {

    // MARK: - String Similarity

    /// Compute string similarity using Levenshtein distance
    static func stringSimilarity(_ a: String, _ b: String) -> Float {
        let aLower = a.lowercased()
        let bLower = b.lowercased()

        let distance = StringUtils.levenshteinDistance(aLower, bLower)
        let maxLen = max(aLower.count, bLower.count)

        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Compute string similarity with length penalty for compound matches.
    /// Penalizes when compound length differs significantly from vocab term length.
    static func lengthPenalizedSimilarity(_ compound: String, _ vocabTerm: String) -> Float {
        let baseSimilarity = stringSimilarity(compound, vocabTerm)

        // Length ratio: how well do the lengths match?
        let compoundLen = Float(compound.count)
        let vocabLen = Float(vocabTerm.count)
        let lengthRatio = min(compoundLen, vocabLen) / max(compoundLen, vocabLen)

        // Apply square root to soften the penalty
        return baseSimilarity * sqrt(lengthRatio)
    }

    // MARK: - Normalized Forms

    /// Represents a normalized form of a vocabulary term (canonical or alias)
    struct NormalizedForm: Hashable {
        let normalized: String
        let wordCount: Int
    }

    /// Build all normalized forms (canonical + aliases) for a vocabulary term
    func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        normalizedFormsByTermLower[term.textLowercased] ?? []
    }

    static func buildNormalizedFormCache(for terms: [CustomVocabularyTerm]) -> [String: [NormalizedForm]] {
        var rawFormsByTermLower: [String: [String]] = [:]
        for term in terms {
            rawFormsByTermLower[term.textLowercased, default: []].append(term.text)
            if let aliases = term.aliases {
                rawFormsByTermLower[term.textLowercased, default: []].append(contentsOf: aliases)
            }
        }

        var cache: [String: [NormalizedForm]] = [:]
        cache.reserveCapacity(rawFormsByTermLower.count)
        for (termLower, rawForms) in rawFormsByTermLower {
            var seen = Set<String>()
            var forms: [NormalizedForm] = []
            forms.reserveCapacity(rawForms.count)
            for raw in rawForms {
                let normalized = normalizeForSimilarity(raw)
                guard !normalized.isEmpty else { continue }
                guard seen.insert(normalized).inserted else { continue }
                let wordCount = normalized.split(separator: " ").count
                forms.append(NormalizedForm(normalized: normalized, wordCount: wordCount))
            }
            cache[termLower] = forms
        }
        return cache
    }

    /// Build normalized component words from every multi-word vocabulary form.
    ///
    /// Used to prevent broad keyword lists from replacing a component of a
    /// known multi-word term with an unrelated single-word distractor, e.g.
    /// `Aaron` → `Atryn` when `Dr. Aaron Petrov` is also present.
    func buildMultiWordVocabularyComponentSet() -> Set<String> {
        multiWordVocabularyComponentSet
    }

    static func buildMultiWordVocabularyComponentSet(
        from normalizedFormsByTermLower: [String: [NormalizedForm]]
    ) -> Set<String> {
        var components = Set<String>()
        for forms in normalizedFormsByTermLower.values {
            for form in forms where form.wordCount > 1 {
                let words = form.normalized.split(separator: " ").map(String.init)
                components.formUnion(words)
            }
        }
        return components
    }

    // MARK: - Similarity Thresholds

    /// Determine required similarity threshold based on span length and word length
    /// Note: Using permissive thresholds to avoid rejecting valid matches
    func requiredSimilarity(minSimilarity: Float, spanLength: Int) -> Float {
        // Multi-word spans: slightly higher threshold to avoid false positives
        if spanLength >= 2 {
            return max(minSimilarity, 0.55)
        }

        // Single words: use the configured minimum similarity
        // Note: The 0.85 threshold for short words was too aggressive (caused regression)
        return minSimilarity
    }

    /// Multi-word replacement candidates should be anchored at one edge of a
    /// vocabulary form. This prevents spans like `to Dr. Felix Quinones` or
    /// `Dr. Felix Quinones reviewed` from replacing and deleting the leading
    /// or trailing non-vocabulary word.
    func multiWordSpanHasAnchoredEdge(spanWords: [String], forms: [NormalizedForm]) -> Bool {
        guard let first = spanWords.first, let last = spanWords.last else { return false }
        for form in forms {
            let words = form.normalized.split(separator: " ").map(String.init)
            guard let formFirst = words.first, let formLast = words.last else { continue }
            if spanWords.count > words.count {
                if first == formFirst && last == formLast { return true }
                continue
            }
            if first == formFirst || last == formLast { return true }
        }
        return false
    }

    /// True when a shortened source span is only missing a vocab edge word
    /// that is already present immediately next to the span in the transcript.
    ///
    /// Example: source span `Aaron Petrov` should not replace with
    /// `Dr. Aaron Petrov` if the previous source word is already `Dr`, or the
    /// output becomes `Dr. Dr. Aaron Petrov`.
    func spanHasAdjacentOmittedVocabEdge(
        spanWords: [String],
        previousWord: String?,
        nextWord: String?,
        forms: [NormalizedForm]
    ) -> Bool {
        guard !spanWords.isEmpty else { return false }
        for form in forms {
            let words = form.normalized.split(separator: " ").map(String.init)
            guard words.count > spanWords.count else { continue }

            if words.suffix(spanWords.count).elementsEqual(spanWords),
                let previousWord,
                previousWord == words[words.count - spanWords.count - 1]
            {
                return true
            }

            if words.prefix(spanWords.count).elementsEqual(spanWords),
                let nextWord,
                nextWord == words[spanWords.count]
            {
                return true
            }
        }
        return false
    }

    // MARK: - Text Utilities

    /// Preserve capitalization from original word in replacement
    func preserveCapitalization(original: String, replacement: String) -> String {
        guard let firstChar = original.first else { return replacement }

        let trailingPunctuation = original.reversed().prefix { char in
            char.isPunctuation && char != "-" && char != "'"
        }.reversed()
        let suffix = String(trailingPunctuation)
        let replacementWithPunctuation =
            suffix.isEmpty || replacement.hasSuffix(suffix) ? replacement : replacement + suffix

        let replacementHasIntentionalCasing = replacement.contains { $0.isUppercase }
        if replacementHasIntentionalCasing {
            return replacementWithPunctuation
        }

        if firstChar.isUppercase && replacementWithPunctuation.first?.isLowercase == true {
            return replacementWithPunctuation.prefix(1).uppercased() + replacementWithPunctuation.dropFirst()
        }
        return replacementWithPunctuation
    }

    /// Normalize text for similarity checks: lowercase, collapse whitespace,
    /// and strip punctuation while preserving letters, numbers, apostrophes, and hyphens.
    static func normalizeForSimilarity(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'-"))
        var result = ""
        var lastWasSpace = true

        for scalar in text.lowercased().unicodeScalars {
            if allowed.contains(scalar) {
                result.append(Character(scalar))
                lastWasSpace = false
            } else if scalar == " " || scalar == "\t" || scalar == "\n" {
                if !lastWasSpace && !result.isEmpty {
                    result.append(" ")
                    lastWasSpace = true
                }
            }
            // Skip other characters (punctuation)
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Build set of normalized vocabulary terms for guard checks
    func buildVocabularyNormalizedSet() -> Set<String> {
        vocabularyNormalizedSet
    }

    static func buildVocabularyNormalizedSet(from normalizedFormsByTermLower: [String: [NormalizedForm]]) -> Set<String>
    {
        var normalizedSet = Set<String>()
        for forms in normalizedFormsByTermLower.values {
            for form in forms {
                normalizedSet.insert(form.normalized)
            }
        }
        return normalizedSet
    }
}

// MARK: - Token Word Boundary Utilities

/// Check if a token string indicates a word boundary.
///
/// SentencePiece and TDT tokenizers use prefixes to indicate word starts:
/// - `▁` (U+2581 LOWER ONE EIGHTH BLOCK) - SentencePiece convention
/// - ` ` (space) - TDT/some tokenizer formats
///
/// - Parameter token: The token string to check
/// - Returns: True if the token starts a new word
public func isWordBoundary(_ token: String) -> Bool {
    token.hasPrefix(ASRConstants.sentencePieceWordBoundary) || token.hasPrefix(" ")
}

/// Strip word boundary prefix from a token.
///
/// Removes the leading `▁` or space character if present.
///
/// - Parameter token: The token string to process
/// - Returns: Token with word boundary prefix removed
public func stripWordBoundaryPrefix(_ token: String) -> String {
    if token.hasPrefix(ASRConstants.sentencePieceWordBoundary) || token.hasPrefix(" ") {
        return String(token.dropFirst())
    }
    return token
}
