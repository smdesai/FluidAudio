import Foundation
import OSLog

/// A chunk of text prepared for synthesis.
/// - words: The original words in this chunk
/// - phonemes: Flat phoneme sequence with single-space separators between words
/// - totalFrames: Reserved for legacy/frame-aware modes (unused here)
/// - pauseAfterMs: Silence to insert after this chunk (punctuation/paragraph driven)
struct TextChunk {
    let words: [String]
    let phonemes: [String]
    let totalFrames: Float
    let pauseAfterMs: Int
}

/// Punctuation-aware chunker that splits paragraphs → sentences → clauses
/// and packs them under the model token budget.
enum KokoroChunker {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroChunker")
    /// Build chunks under the token budget.
    /// - Parameters:
    ///   - text: Raw input text
    ///   - wordToPhonemes: Mapping of lowercase words to phoneme arrays
    ///   - targetTokens: Model token capacity for `input_ids`
    ///   - hasLanguageToken: Whether to reserve one token for a language marker
    /// - Returns: Array of TextChunk with punctuation-driven `pauseAfterMs`
    static func chunk(
        text: String,
        wordToPhonemes: [String: [String]],
        targetTokens: Int,
        hasLanguageToken: Bool
    ) -> [TextChunk] {
        let baseOverhead = 2 + (hasLanguageToken ? 1 : 0)  // BOS/EOS + optional language token
        // Safety margin to avoid hard edge near model capacity (prevents mid-sentence cut-offs)
        let safety = 12
        let cap = max(1, targetTokens - safety)

        // Pause mapping (ms)
        let pauseSentence = 300
        let pauseClause = 150
        let pauseParagraph = 500

        func phonemizeWords(_ words: [String]) -> [String] {
            let vocab = KokoroVocabulary.getVocabulary()
            let allowed = Set(vocab.keys)

            func isPunct(_ ch: Character) -> Bool {
                return !(ch.isLetter || ch.isNumber || ch.isWhitespace || ch == "'")
            }
            func normalize(_ s: String) -> String {
                let lowered = s.lowercased()
                    .replacingOccurrences(of: "\u{2019}", with: "'")
                    .replacingOccurrences(of: "\u{2018}", with: "'")
                let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
                return String(lowered.unicodeScalars.filter { allowedSet.contains($0) })
            }

            var out: [String] = []
            for (wi, w) in words.enumerated() {
                // Split w into runs: word segments and single punctuation chars
                var seg = ""
                func flushSeg() {
                    guard !seg.isEmpty else { return }
                    let key = normalize(seg)
                    if let arr = wordToPhonemes[key] {
                        out.append(contentsOf: arr)
                    } else {
                        // Use C eSpeak NG for OOV phonemization only (if available)
                        #if canImport(CEspeakNG)
                        if let ipa = EspeakG2P.shared.phonemize(word: key) {
                            let mapped = PhonemeMapper.mapIPA(ipa, allowed: allowed)
                            if !mapped.isEmpty {
                                print(
                                    "[G2P] word=\(key) | ipa=\(ipa.joined(separator: " ")) | map=\(mapped.joined(separator: " "))"
                                )
                                out.append(contentsOf: mapped)
                                KokoroChunker.logger.info("EspeakG2P used for OOV word: \(key)")
                            } else {
                                print("[G2P] word=\(key) | ipa=<none> | map=<empty>")
                                KokoroChunker.logger.warning("OOV word yielded no mappable IPA tokens: \(key)")
                            }
                        } else {
                            print("[G2P] word=\(key) | ipa=<failed> | map=<empty>")
                            KokoroChunker.logger.warning("EspeakG2P failed for OOV word: \(key)")
                        }
                        #else
                        // G2P not available in this build; skip OOV word
                        KokoroChunker.logger.warning("CEspeakNG not available; skipping OOV word: \(key)")
                        #endif
                    }
                    seg.removeAll()
                }
                for ch in w {
                    if isPunct(ch) {
                        flushSeg()
                        let p = String(ch)
                        if allowed.contains(p) { out.append(p) }
                    } else {
                        seg.append(ch)
                    }
                }
                flushSeg()
                if wi != words.count - 1 { out.append(" ") }
            }
            if out.last == " " { out.removeLast() }
            return out
        }

        func splitSentences(_ paragraph: String) -> [(String, Character?)] {
            var units: [(String, Character?)] = []
            var buf = ""
            for ch in paragraph {
                buf.append(ch)
                if ch == "." || ch == "!" || ch == "?" {
                    units.append((buf.trimmingCharacters(in: .whitespaces), ch))
                    buf = ""
                }
            }
            let tail = buf.trimmingCharacters(in: .whitespaces)
            if !tail.isEmpty { units.append((tail, nil)) }
            return units
        }

        func splitClauses(_ sentence: String, endPunct: Character?) -> [(String, Int)] {
            var parts: [(String, Int)] = []
            var buf = ""
            for ch in sentence {
                if ch == "," || ch == ";" || ch == ":" {
                    let trimmed = buf.trimmingCharacters(in: .whitespaces)
                    if !trimmed.isEmpty { parts.append((trimmed, pauseClause)) }
                    buf = ""
                } else {
                    buf.append(ch)
                }
            }
            let last = buf.trimmingCharacters(in: .whitespaces)
            if !last.isEmpty { parts.append((last, endPunct != nil ? pauseSentence : 0)) }
            return parts
        }

        let paragraphs = text.components(separatedBy: "\n\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }

        var chunks: [TextChunk] = []
        for (pIndex, para) in paragraphs.enumerated() {
            let sentences = splitSentences(para)
            var units: [(words: [String], pause: Int)] = []
            for (sent, endP) in sentences {
                let clauses = splitClauses(sent, endPunct: endP)
                for (cl, pause) in clauses {
                    let words = cl.lowercased().split(separator: " ").map { String($0) }
                    if !words.isEmpty { units.append((words, pause)) }
                }
            }

            var curWords: [String] = []
            var curPhon: [String] = []
            var curTokenCount = baseOverhead
            var lastPause = pIndex < paragraphs.count - 1 ? pauseParagraph : 0

            for (words, pause) in units {
                let ph = phonemizeWords(words)
                // If the entire unit doesn't fit into an empty chunk, split by words to respect budget.
                let fitsEmpty = (baseOverhead + ph.count) <= cap

                if ph.isEmpty {
                    // Unit has no known phonemes; treat as zero-cost. Skip, but carry over pause.
                    lastPause = pause
                    continue
                }

                if fitsEmpty == false {
                    // Micro-split: accumulate word-by-word within this unit.
                    var subWords: [String] = []
                    var subPhon: [String] = []
                    var subCount = baseOverhead

                    func flushSub(finalPause: Int) {
                        if !subPhon.isEmpty {
                            if subPhon.last == " " { subPhon.removeLast() }
                            chunks.append(
                                TextChunk(words: subWords, phonemes: subPhon, totalFrames: 0, pauseAfterMs: finalPause))
                            subWords.removeAll(keepingCapacity: true)
                            subPhon.removeAll(keepingCapacity: true)
                            subCount = baseOverhead
                        }
                    }

                    for (i, w) in words.enumerated() {
                        let wPh = phonemizeWords([w])
                        // Fallback cost for unknown words: treat as small cost so we still split; no phonemes appended.
                        let wCost = (wPh.isEmpty ? 1 : wPh.count) + (subPhon.isEmpty ? 0 : 1)
                        if subCount + wCost <= cap {
                            if !wPh.isEmpty {
                                if !subPhon.isEmpty { subPhon.append(" ") }
                                subPhon.append(contentsOf: wPh)
                            }
                            subWords.append(w)
                            subCount += wCost
                        } else {
                            // flush with no internal pause between sub-chunks
                            flushSub(finalPause: 0)
                            if !wPh.isEmpty {
                                subPhon.append(contentsOf: wPh)
                            }
                            subWords.append(w)
                            subCount = baseOverhead + (wPh.isEmpty ? 1 : wPh.count)
                        }
                        // At end of unit, flush with the unit's pause
                        if i == words.count - 1 {
                            flushSub(finalPause: pause)
                            lastPause = pause
                        }
                    }
                    continue
                }

                // Normal packing at unit granularity
                let additional = ph.count + (curPhon.isEmpty ? 0 : 1)  // +1 for join space
                if curTokenCount + additional <= cap {
                    if !curPhon.isEmpty { curPhon.append(" ") }
                    curPhon.append(contentsOf: ph)
                    curWords.append(contentsOf: words)
                    curTokenCount += additional
                    lastPause = pause
                } else {
                    if !curPhon.isEmpty {
                        if curPhon.last == " " { curPhon.removeLast() }
                        chunks.append(
                            TextChunk(words: curWords, phonemes: curPhon, totalFrames: 0, pauseAfterMs: lastPause))
                    }
                    curWords = words
                    curPhon = ph
                    curTokenCount = baseOverhead + ph.count
                    lastPause = pause
                }
            }
            if !curPhon.isEmpty {
                if curPhon.last == " " { curPhon.removeLast() }
                let paraPause = (pIndex < paragraphs.count - 1) ? pauseParagraph : lastPause
                chunks.append(TextChunk(words: curWords, phonemes: curPhon, totalFrames: 0, pauseAfterMs: paraPause))
            }
        }
        return chunks
    }
}
