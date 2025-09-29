import Foundation

#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// Lightweight chunk representation passed into Kokoro synthesis.
struct TextChunk: Sendable {
    let words: [String]
    let atoms: [String]
    let phonemes: [String]
    let totalFrames: Float
    let pauseAfterMs: Int
    let text: String
}

/// Chunker that mirrors the reference MLX Swift sentence-merging strategy for English text.
enum KokoroChunker {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroChunker")
    private static let decimalDigits = CharacterSet.decimalDigits
    private static let apostropheCharacters: Set<Character> = ["'", "’", "ʼ", "‛", "‵", "′"]
    private struct ChunkSegment {
        let text: String
        let range: Range<String.Index>
    }
    /// Public entry point used by `KokoroSynthesizer`
    static func chunk(
        text: String,
        wordToPhonemes: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        targetTokens: Int,
        hasLanguageToken: Bool,
        allowedPhonemes: Set<String>
    ) -> [TextChunk] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let collapsed = collapseNewlines(trimmed)
        let preprocessResult = KokoroTextPreprocessor().preprocess(collapsed)
        let normalized = preprocessResult.text

        let (sentences, _) = splitIntoSentences(normalized)
        guard !sentences.isEmpty else { return [] }

        let segmentsByPeriods = applyRefinements(sentences)
        guard !segmentsByPeriods.isEmpty else {
            logger.info("Kokoro chunker produced no segments after refinement")
            return []
        }

        let capacity = computeCapacity(targetTokens: targetTokens, hasLanguageToken: hasLanguageToken)

        for (index, segment) in segmentsByPeriods.enumerated() {
            logger.info("segmentsByPeriods[\(index)]: \(segment.text)")
            let tokenCount = tokenCountForSegment(
                segment,
                lexicon: wordToPhonemes,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowedPhonemes,
                capacity: capacity,
                preprocessResult: preprocessResult,
                fullText: normalized
            )
            logger.debug("segmentsByPeriods[\(index)] tokenCount=\(tokenCount) capacity=\(capacity)")
        }

        var segmentsByPunctuations: [ChunkSegment] = []
        segmentsByPunctuations.reserveCapacity(segmentsByPeriods.count)

        for (periodIndex, segment) in segmentsByPeriods.enumerated() {
            let count = tokenCountForSegment(
                segment,
                lexicon: wordToPhonemes,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowedPhonemes,
                capacity: capacity,
                preprocessResult: preprocessResult,
                fullText: normalized
            )

            if count > capacity {
                let fragments = splitSegmentByPunctuation(segment, in: normalized)
                let reassembled = reassembleFragments(
                    fragments,
                    fullText: normalized,
                    preprocessResult: preprocessResult,
                    lexicon: wordToPhonemes,
                    caseSensitiveLexicon: caseSensitiveLexicon,
                    allowed: allowedPhonemes,
                    capacity: capacity
                )
                if !reassembled.isEmpty {
                    logger.debug(
                        "Segment exceeded capacity; punctuation split yielded \(reassembled.count) subsegments"
                    )
                    logger.info(
                        "segmentsByPeriodsSplit[\(periodIndex)]: original='\(segment.text)'"
                    )
                    for (fragmentIndex, part) in reassembled.enumerated() {
                        logger.info(
                            "segmentsByPeriodsSplit[\(periodIndex)].part[\(fragmentIndex)]: \(part.text)"
                        )
                    }
                    segmentsByPunctuations.append(contentsOf: reassembled)
                    continue
                }
                logger.warning(
                    "segmentsByPeriodsSplit[\(periodIndex)]: no punctuation-based split within capacity; deferring to chunk builder"
                )
            }

            segmentsByPunctuations.append(segment)
        }

        for (index, segment) in segmentsByPunctuations.enumerated() {
            logger.info("segmentsByPunctuations[\(index)]: \(segment.text)")
        }

        let chunks = segmentsByPunctuations.flatMap { segment in
            buildChunks(
                from: segment,
                fullText: normalized,
                preprocessResult: preprocessResult,
                lexicon: wordToPhonemes,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowedPhonemes,
                capacity: capacity
            )
        }

        return chunks
    }

    private static func computeCapacity(targetTokens: Int, hasLanguageToken: Bool) -> Int {
        let baseOverhead = 2 + (hasLanguageToken ? 1 : 0)
        let safety = 12
        return max(1, targetTokens - baseOverhead - safety)
    }

    // MARK: - Sentence Processing

    private static func splitIntoSentences(_ text: String) -> ([ChunkSegment], NLLanguage?) {
        #if canImport(NaturalLanguage)
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        let dominant = recognizer.dominantLanguage ?? .english

        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.setLanguage(dominant)

        var sentences: [ChunkSegment] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            if let trimmed = trimmedRange(range, in: text) {
                let segmentText = String(text[trimmed])
                sentences.append(ChunkSegment(text: segmentText, range: trimmed))
            }
            return true
        }

        if sentences.isEmpty, let entireRange = trimmedRange(text.startIndex..<text.endIndex, in: text) {
            return ([ChunkSegment(text: String(text[entireRange]), range: entireRange)], dominant)
        }
        return (sentences, dominant)
        #else
        let delimiters = CharacterSet(charactersIn: ".?!")
        var sentences: [ChunkSegment] = []
        var currentStart = text.startIndex
        var index = currentStart

        while index < text.endIndex {
            let character = text[index]
            index = text.index(after: index)
            if let scalar = character.unicodeScalars.first, delimiters.contains(scalar) {
                let sentenceRange = currentStart..<index
                if let trimmed = trimmedRange(sentenceRange, in: text) {
                    sentences.append(ChunkSegment(text: String(text[trimmed]), range: trimmed))
                }
                currentStart = index
            }
        }

        if currentStart < text.endIndex {
            let tailRange = currentStart..<text.endIndex
            if let trimmed = trimmedRange(tailRange, in: text) {
                sentences.append(ChunkSegment(text: String(text[trimmed]), range: trimmed))
            }
        }

        if sentences.isEmpty, let entireRange = trimmedRange(text.startIndex..<text.endIndex, in: text) {
            return ([ChunkSegment(text: String(text[entireRange]), range: entireRange)], nil)
        }

        return (sentences, nil)
        #endif
    }

    private static func applyRefinements(_ sentences: [ChunkSegment]) -> [ChunkSegment] {
        sentences.filter { !$0.text.isEmpty }
    }

    private static func trimmedRange(_ range: Range<String.Index>, in text: String) -> Range<String.Index>? {
        var lower = range.lowerBound
        var upper = range.upperBound

        while lower < upper, text[lower].isWhitespace {
            lower = text.index(after: lower)
        }

        while upper > lower {
            let before = text.index(before: upper)
            if text[before].isWhitespace {
                upper = before
            } else {
                break
            }
        }

        return lower < upper ? lower..<upper : nil
    }

    // MARK: - Chunk Construction

    private static func buildChunks(
        from segment: ChunkSegment,
        fullText: String,
        preprocessResult: KokoroTextPreprocessor.Result,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        capacity: Int
    ) -> [TextChunk] {
        let atoms = tokenizeAtoms(in: segment, fullText: fullText, preprocessResult: preprocessResult)
        guard !atoms.isEmpty else { return [] }

        var chunks: [TextChunk] = []
        var chunkWords: [String] = []
        var chunkAtoms: [String] = []
        var chunkPhonemes: [String] = []
        var chunkTokenCount = 0
        var needsWordSeparator = false
        var missing: Set<String> = []

        func flushChunk() {
            guard !chunkPhonemes.isEmpty else { return }
            if chunkPhonemes.last == " " {
                chunkPhonemes.removeLast()
                chunkTokenCount -= 1
            }
            let textValue = chunkAtoms.reduce(into: "") { partial, atom in
                partial = appendSegment(partial, with: atom)
            }.trimmingCharacters(in: .whitespacesAndNewlines)
            chunks.append(
                TextChunk(
                    words: chunkWords,
                    atoms: chunkAtoms,
                    phonemes: chunkPhonemes,
                    totalFrames: 0,
                    pauseAfterMs: 0,
                    text: textValue
                )
            )
            chunkWords.removeAll(keepingCapacity: true)
            chunkAtoms.removeAll(keepingCapacity: true)
            chunkPhonemes.removeAll(keepingCapacity: true)
            chunkTokenCount = 0
            needsWordSeparator = false
        }

        for atom in atoms {
            if atom.suppress {
                switch atom.kind {
                case .word:
                    chunkWords.append(atom.text)
                    chunkAtoms.append(atom.text)
                    needsWordSeparator = true
                case .punctuation:
                    chunkAtoms.append(atom.text)
                    needsWordSeparator = false
                }
                continue
            }

            switch atom.kind {
            case .word(let original):
                guard
                    let resolved = phonemesForAtom(
                        atom,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &missing
                    )
                else {
                    continue
                }

                var tokenCost = resolved.count
                if needsWordSeparator {
                    tokenCost += 1
                }

                if chunkTokenCount + tokenCost > capacity && !chunkPhonemes.isEmpty {
                    flushChunk()
                }

                if needsWordSeparator {
                    chunkPhonemes.append(" ")
                    chunkTokenCount += 1
                }

                chunkPhonemes.append(contentsOf: resolved)
                chunkTokenCount += resolved.count
                chunkWords.append(original)
                chunkAtoms.append(original)
                needsWordSeparator = true

            case .punctuation(let symbol):
                guard
                    let resolved = phonemesForAtom(
                        atom,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &missing
                    )
                else {
                    continue
                }

                if chunkTokenCount + resolved.count > capacity && !chunkPhonemes.isEmpty {
                    flushChunk()
                }

                chunkPhonemes.append(contentsOf: resolved)
                chunkTokenCount += resolved.count
                chunkAtoms.append(symbol)
                needsWordSeparator = false
            }
        }

        flushChunk()

        if !missing.isEmpty {
            logger.warning("Missing phoneme entries for: \(missing.sorted().joined(separator: ", "))")
        }

        return chunks
    }

    private enum AtomKind {
        case word(String)
        case punctuation(String)
    }

    private struct AtomToken {
        let text: String
        let kind: AtomKind
        let range: Range<String.Index>
        let feature: KokoroTextPreprocessor.Result.FeatureRange.Feature?
        let suppress: Bool
    }

    private static func tokenizeAtoms(
        in segment: ChunkSegment,
        fullText: String,
        preprocessResult: KokoroTextPreprocessor.Result
    ) -> [AtomToken] {
        var atoms: [AtomToken] = []
        var currentWord = ""
        var currentWordStart: String.Index?

        func flushWord(upTo end: String.Index) {
            guard let start = currentWordStart, !currentWord.isEmpty else { return }
            let range = start..<end
            let annotation = preprocessResult.annotation(for: range)
            atoms.append(
                AtomToken(
                    text: currentWord,
                    kind: .word(currentWord),
                    range: range,
                    feature: annotation.feature,
                    suppress: annotation.suppress
                )
            )
            currentWord.removeAll(keepingCapacity: true)
            currentWordStart = nil
        }

        var index = segment.range.lowerBound
        while index < segment.range.upperBound {
            let character = fullText[index]

            if character.isWhitespace {
                flushWord(upTo: index)
                index = fullText.index(after: index)
                continue
            }

            if character.isLetter || character.isNumber || apostropheCharacters.contains(character) {
                if currentWordStart == nil {
                    currentWordStart = index
                }
                currentWord.append(apostropheCharacters.contains(character) ? "'" : character)
                index = fullText.index(after: index)
                continue
            }

            flushWord(upTo: index)
            let nextIndex = fullText.index(after: index)
            let range = index..<nextIndex
            let annotation = preprocessResult.annotation(for: range)
            atoms.append(
                AtomToken(
                    text: String(character),
                    kind: .punctuation(String(character)),
                    range: range,
                    feature: annotation.feature,
                    suppress: annotation.suppress
                )
            )
            index = nextIndex
        }

        flushWord(upTo: segment.range.upperBound)
        return atoms
    }

    private static func resolvePhonemes(
        for original: String,
        normalized: String,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        missing: inout Set<String>
    ) -> [String]? {
        var phonemes = caseSensitiveLexicon[original]

        if phonemes == nil, let exactNormalized = caseSensitiveLexicon[normalized] {
            phonemes = exactNormalized
        }

        if phonemes == nil {
            phonemes = lexicon[normalized]
        }

        #if canImport(ESpeakNG)
        if phonemes == nil {
            if #available(macOS 13.0, iOS 16.0, *) {
                if let ipa = EspeakG2P.shared.phonemize(word: normalized) {
                    let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                    if !mapped.isEmpty {
                        phonemes = mapped
                    }
                }
            }
        }
        #endif

        if phonemes == nil,
            let spelledTokens = spelledOutTokens(for: normalized),
            !spelledTokens.isEmpty
        {
            var spelledPhonemes: [String] = []
            var success = true
            var firstSegment = true
            for spelled in spelledTokens {
                var segment = lexicon[spelled]

                #if canImport(ESpeakNG)
                if segment == nil {
                    if #available(macOS 13.0, iOS 16.0, *) {
                        if let ipa = EspeakG2P.shared.phonemize(word: spelled) {
                            let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                            if !mapped.isEmpty {
                                segment = mapped
                            }
                        }
                    }
                }
                #endif

                if segment == nil, let fallback = letterPronunciations[spelled] {
                    let filtered = fallback.filter { allowed.contains($0) }
                    if !filtered.isEmpty {
                        segment = filtered
                    }
                }

                guard var resolvedSegment = segment, !resolvedSegment.isEmpty else {
                    success = false
                    break
                }

                resolvedSegment = resolvedSegment.filter { allowed.contains($0) }
                if resolvedSegment.isEmpty {
                    success = false
                    break
                }

                if !firstSegment {
                    spelledPhonemes.append(" ")
                }
                spelledPhonemes.append(contentsOf: resolvedSegment)
                firstSegment = false
            }

            if success, !spelledPhonemes.isEmpty {
                phonemes = spelledPhonemes
            }
        }

        if phonemes == nil, let fallback = letterPronunciations[normalized] {
            let filtered = fallback.filter { allowed.contains($0) }
            if !filtered.isEmpty {
                phonemes = filtered
            }
        }

        guard var resolved = phonemes, !resolved.isEmpty else {
            missing.insert(normalized)
            return nil
        }

        resolved = resolved.filter { allowed.contains($0) }
        guard !resolved.isEmpty else {
            missing.insert(normalized)
            return nil
        }

        return resolved
    }

    private static func tokenCountForSegment(
        _ segment: ChunkSegment,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        capacity: Int,
        preprocessResult: KokoroTextPreprocessor.Result,
        fullText: String
    ) -> Int {
        let atoms = tokenizeAtoms(in: segment, fullText: fullText, preprocessResult: preprocessResult)
        guard !atoms.isEmpty else { return 0 }

        var dummyMissing: Set<String> = []
        var tokenCount = 0
        var needsWordSeparator = false

        for atom in atoms {
            if atom.suppress { continue }

            switch atom.kind {
            case .word, .punctuation:
                guard
                    let phonemes = phonemesForAtom(
                        atom,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &dummyMissing
                    )
                else {
                    continue
                }
                tokenCount += phonemes.count
                if case .word = atom.kind {
                    if needsWordSeparator {
                        tokenCount += 1
                    }
                    needsWordSeparator = true
                } else {
                    needsWordSeparator = false
                }
            }

            if tokenCount > capacity {
                return tokenCount
            }
        }

        return tokenCount
    }

    private static func reassembleFragments(
        _ fragments: [ChunkSegment],
        fullText: String,
        preprocessResult: KokoroTextPreprocessor.Result,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        capacity: Int
    ) -> [ChunkSegment] {
        guard !fragments.isEmpty else { return [] }

        var assembled: [ChunkSegment] = []
        var currentRange = fragments[0].range

        func flushCurrent() -> Bool {
            guard let trimmed = trimmedRange(currentRange, in: fullText) else { return true }
            let segment = ChunkSegment(text: String(fullText[trimmed]), range: trimmed)
            let count = tokenCountForSegment(
                segment,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowed,
                capacity: capacity,
                preprocessResult: preprocessResult,
                fullText: fullText
            )
            if count > capacity {
                return false
            }
            assembled.append(segment)
            return true
        }

        for fragment in fragments.dropFirst() {
            let candidateRange = currentRange.lowerBound..<fragment.range.upperBound
            let candidateSegment = ChunkSegment(text: String(fullText[candidateRange]), range: candidateRange)
            let candidateTokens = tokenCountForSegment(
                candidateSegment,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowed,
                capacity: capacity,
                preprocessResult: preprocessResult,
                fullText: fullText
            )

            if candidateTokens <= capacity {
                currentRange = candidateRange
            } else {
                if !flushCurrent() {
                    return []
                }
                currentRange = fragment.range
            }
        }

        if !flushCurrent() {
            return []
        }

        return assembled
    }

    private static func phonemesForAtom(
        _ atom: AtomToken,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        missing: inout Set<String>
    ) -> [String]? {
        switch atom.kind {
        case .word(let original):
            let normalized = normalize(original)
            guard !normalized.isEmpty else { return nil }

            var overridePhonemes: [String]? = nil
            if let feature = atom.feature {
                switch feature {
                case .phoneme(let override):
                    overridePhonemes = manualPhonemes(override, allowed: allowed)
                case .alias(let aliasText):
                    overridePhonemes = phonemesForAlias(
                        aliasText,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &missing
                    )
                }
            }

            if let override = overridePhonemes, !override.isEmpty {
                return override
            }

            return resolvePhonemes(
                for: original,
                normalized: normalized,
                lexicon: lexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                allowed: allowed,
                missing: &missing
            )

        case .punctuation(let symbol):
            guard allowed.contains(symbol) else { return nil }
            return [symbol]
        }
    }

    private static func manualPhonemes(
        _ override: String,
        allowed: Set<String>
    ) -> [String]? {
        let trimmed = override.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let graphemeTokens = trimmed.compactMap { character -> String? in
            character.isWhitespace ? nil : String(character)
        }
        let mapped = PhonemeMapper.mapIPA(graphemeTokens, allowed: allowed)
        if !mapped.isEmpty {
            return mapped
        }

        let parts = trimmed.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        if parts.isEmpty {
            return allowed.contains(trimmed) ? [trimmed] : nil
        }

        let filtered = parts.filter { allowed.contains($0) }
        return filtered.count == parts.count ? filtered : nil
    }

    private static func phonemesForAlias(
        _ alias: String,
        lexicon: [String: [String]],
        caseSensitiveLexicon: [String: [String]],
        allowed: Set<String>,
        missing: inout Set<String>
    ) -> [String]? {
        let tokens = tokenizeAlias(alias)
        guard !tokens.isEmpty else { return nil }

        var aliasPhonemes: [String] = []
        var needsSeparator = false

        for token in tokens {
            switch token {
            case .word(let word):
                let normalized = normalize(word)
                guard !normalized.isEmpty else { continue }
                guard
                    let resolved = resolvePhonemes(
                        for: word,
                        normalized: normalized,
                        lexicon: lexicon,
                        caseSensitiveLexicon: caseSensitiveLexicon,
                        allowed: allowed,
                        missing: &missing
                    )
                else {
                    return nil
                }
                if needsSeparator {
                    aliasPhonemes.append(" ")
                }
                aliasPhonemes.append(contentsOf: resolved)
                needsSeparator = true
            case .punctuation(let symbol):
                guard allowed.contains(symbol) else { continue }
                aliasPhonemes.append(symbol)
                needsSeparator = false
            }
        }

        return aliasPhonemes.isEmpty ? nil : aliasPhonemes
    }

    private static func tokenizeAlias(_ text: String) -> [AtomKind] {
        var tokens: [AtomKind] = []
        var currentWord = ""

        func flushWord() {
            guard !currentWord.isEmpty else { return }
            tokens.append(.word(currentWord))
            currentWord.removeAll(keepingCapacity: true)
        }

        for character in text {
            if character.isWhitespace {
                flushWord()
                continue
            }

            if character.isLetter || character.isNumber || apostropheCharacters.contains(character) {
                currentWord.append(apostropheCharacters.contains(character) ? "'" : character)
            } else {
                flushWord()
                tokens.append(.punctuation(String(character)))
            }
        }

        flushWord()
        return tokens
    }

    private static func splitSegmentByPunctuation(_ segment: ChunkSegment, in fullText: String) -> [ChunkSegment] {
        guard !segment.text.isEmpty else { return [] }

        var segments: [ChunkSegment] = []
        var currentStart = segment.range.lowerBound
        let breakCharacters = CharacterSet(charactersIn: ",;:")
        let separatorTokens = [": ", "; ", ", "]

        var index = currentStart
        while index < segment.range.upperBound {
            let character = fullText[index]
            if let scalar = character.unicodeScalars.first, breakCharacters.contains(scalar) {
                var endIndex = fullText.index(after: index)
                for separator in separatorTokens {
                    if endIndex < segment.range.upperBound {
                        let remaining = fullText[endIndex..<segment.range.upperBound]
                        if remaining.hasPrefix(separator) {
                            endIndex =
                                fullText.index(endIndex, offsetBy: separator.count, limitedBy: segment.range.upperBound)
                                ?? segment.range.upperBound
                            break
                        }
                    }
                }

                if let trimmed = trimmedRange(currentStart..<endIndex, in: fullText) {
                    segments.append(ChunkSegment(text: String(fullText[trimmed]), range: trimmed))
                }
                currentStart = endIndex
            }
            index = fullText.index(after: index)
        }

        if currentStart < segment.range.upperBound {
            if let trimmedTail = trimmedRange(currentStart..<segment.range.upperBound, in: fullText) {
                segments.append(ChunkSegment(text: String(fullText[trimmedTail]), range: trimmedTail))
            }
        }

        return segments.isEmpty ? [segment] : segments
    }

    private static func normalize(_ word: String) -> String {
        let lowered = word.lowercased()
        let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
        let filteredScalars = lowered.unicodeScalars.filter { allowedSet.contains($0) }
        return String(String.UnicodeScalarView(filteredScalars))
    }

    private static func collapseNewlines(_ text: String) -> String {
        guard text.contains(where: { $0.isNewline }) else { return text }
        let segments = text.split(whereSeparator: { $0.isNewline })
        return segments.map(String.init).joined(separator: " ")
    }

    private static func appendSegment(_ base: String, with next: String) -> String {
        let trimmedNext = next.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedNext.isEmpty else { return base }
        if base.isEmpty { return trimmedNext }
        if let first = trimmedNext.first, noPrespaceCharacters.contains(first) {
            return base + trimmedNext
        }
        return base + " " + trimmedNext
    }

    private static func spelledOutTokens(for token: String) -> [String]? {
        guard !token.isEmpty else { return nil }
        if token.rangeOfCharacter(from: decimalDigits.inverted) != nil {
            return nil
        }
        guard let value = Int(token) else { return nil }
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.numberStyle = .spellOut
        formatter.maximumFractionDigits = 0
        formatter.roundingMode = .down
        guard let spelled = formatter.string(from: NSNumber(value: value)) else { return nil }
        let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "-"))
        let components =
            spelled
            .lowercased()
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }
        return components.isEmpty ? nil : components
    }

    private static let noPrespaceCharacters: Set<Character> = [
        ",", ";", ":", "!", "?", ".", "…", "—", "–", "'", "\"", ")", "]", "}", "”", "’",
    ]

    private static let letterPronunciations: [String: [String]] = [
        "a": ["e", "ɪ"],
        "b": ["b", "i"],
        "c": ["s", "i"],
        "d": ["d", "i"],
        "e": ["i"],
        "f": ["ɛ", "f"],
        "g": ["ʤ", "i"],
        "h": ["e", "ɪ", "ʧ"],
        "i": ["a", "ɪ"],
        "j": ["ʤ", "e"],
        "k": ["k", "e"],
        "l": ["ɛ", "l"],
        "m": ["ɛ", "m"],
        "n": ["ɛ", "n"],
        "o": ["o"],
        "p": ["p", "i"],
        "q": ["k", "j", "u"],
        "r": ["ɑ", "r"],
        "s": ["ɛ", "s"],
        "t": ["t", "i"],
        "u": ["j", "u"],
        "v": ["v", "i"],
        "w": ["d", "ʌ", "b", "əl", "j", "u"],
        "x": ["ɛ", "k", "s"],
        "y": ["w", "a", "ɪ"],
        "z": ["z", "i"],
    ]
}
