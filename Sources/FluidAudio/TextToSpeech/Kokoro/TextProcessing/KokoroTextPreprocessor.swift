import Foundation

/// Applies Kokoro-specific text normalization prior to chunking.
struct KokoroTextPreprocessor {
    struct Result {
        struct FeatureRange {
            enum Feature {
                case alias(String)
                case phoneme(String)
            }

            let range: Range<String.Index>
            let feature: Feature
        }

        let text: String
        let features: [FeatureRange]

        func annotation(for range: Range<String.Index>) -> (feature: FeatureRange.Feature?, suppress: Bool) {
            for entry in features {
                guard entry.range.lowerBound <= range.lowerBound,
                    range.upperBound <= entry.range.upperBound
                else { continue }

                if entry.range.lowerBound == range.lowerBound {
                    return (entry.feature, false)
                } else {
                    return (nil, true)
                }
            }
            return (nil, false)
        }
    }

    func preprocess(_ text: String) -> Result {
        guard !text.isEmpty else {
            return Result(text: text, features: [])
        }

        var processedText = removeCommasFromNumbers(text)
        processedText = Self.rangeRegex.stringByReplacingMatches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText),
            withTemplate: "$1 to $2"
        )
        processedText = flipMoney(processedText)
        processedText = splitNum(processedText)
        processedText = pointNum(processedText)

        let matches = Self.linkRegex.matches(
            in: processedText,
            range: NSRange(processedText.startIndex..., in: processedText)
        )

        guard !matches.isEmpty else {
            return Result(text: processedText, features: [])
        }

        var result = ""
        var features: [Result.FeatureRange] = []
        var lastEnd = processedText.startIndex

        for match in matches {
            guard let matchRange = Range(match.range, in: processedText) else { continue }
            let leading = processedText[lastEnd..<matchRange.lowerBound]
            result.append(contentsOf: leading)

            guard
                match.numberOfRanges >= 3,
                let originalRange = Range(match.range(at: 1), in: processedText)
            else {
                lastEnd = matchRange.upperBound
                continue
            }

            let original = processedText[originalRange]
            let featureStart = result.endIndex
            result.append(contentsOf: original)
            let featureEnd = result.endIndex

            if let replacementRange = Range(match.range(at: 2), in: processedText) {
                let replacement = String(processedText[replacementRange])
                if let feature = parseFeature(replacement: replacement) {
                    let range = featureStart..<featureEnd
                    features.append(Result.FeatureRange(range: range, feature: feature))
                }
            }

            lastEnd = matchRange.upperBound
        }

        if lastEnd < processedText.endIndex {
            let trailing = processedText[lastEnd..<processedText.endIndex]
            result.append(contentsOf: trailing)
        }

        return Result(text: result, features: features)
    }

    private func parseFeature(replacement: String) -> Result.FeatureRange.Feature? {
        let trimmed = replacement.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if trimmed.hasPrefix("/") && trimmed.hasSuffix("/") && trimmed.count >= 2 {
            let phoneme = String(trimmed.dropFirst().dropLast())
            return .phoneme(phoneme)
        }

        if Float(trimmed) != nil {
            return nil
        }

        return .alias(replacement)
    }

    private func removeCommasFromNumbers(_ text: String) -> String {
        return Self.commaInNumberRegex.stringByReplacingMatches(
            in: text,
            range: NSRange(text.startIndex..., in: text),
            withTemplate: "$1$2$3"
        ).replacingOccurrences(of: ",", with: "")
    }

    private func flipMoney(_ text: String) -> String {
        var result = text
        let matches = Self.currencyRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard let matchRange = Range(match.range, in: text) else { continue }
            let matchText = String(text[matchRange])

            guard let currencySymbol = matchText.first,
                let currency = Self.currencies[currencySymbol]
            else { continue }

            let value = String(matchText.dropFirst())
            let components = value.components(separatedBy: ".")
            let dollars = components[0]
            let cents = components.count > 1 ? components[1] : "0"

            let transformed: String
            if Int(cents) == 0 {
                transformed = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
            } else {
                let dollarPart = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
                transformed = "\(dollarPart) and \(cents) \(currency.cent)s"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func splitNum(_ text: String) -> String {
        var result = text
        let matches = Self.timeRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard let matchRange = Range(match.range, in: text) else { continue }
            let matchText = String(text[matchRange])

            let components = matchText.components(separatedBy: ":")
            guard components.count == 2,
                let hour = Int(components[0]),
                let minute = Int(components[1])
            else { continue }

            let transformed: String
            if minute == 0 {
                transformed = "\(hour) o'clock"
            } else if minute < 10 {
                transformed = "\(hour) oh \(minute)"
            } else {
                transformed = "\(hour) \(minute)"
            }

            result = result.replacingCharacters(in: matchRange, with: "\(transformed)")
        }

        return result
    }

    private func pointNum(_ text: String) -> String {
        var result = text
        let decimalMatches = Self.decimalRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        let linkMatches = Self.linkRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        var excludeRanges: [NSRange] = []
        for linkMatch in linkMatches where linkMatch.numberOfRanges >= 3 {
            excludeRanges.append(linkMatch.range(at: 2))
        }

        for match in decimalMatches.reversed() {
            let matchRange = match.range
            let isExcluded = excludeRanges.contains { NSIntersectionRange(matchRange, $0).length > 0 }
            if isExcluded { continue }

            guard let swiftRange = Range(matchRange, in: text) else { continue }
            let matchText = String(text[swiftRange])

            let components = matchText.components(separatedBy: ".")
            guard components.count == 2 else { continue }

            let integerPart = components[0]
            let decimalDigits = components[1].map { String($0) }.joined(separator: " ")
            let transformed = "\(integerPart) point \(decimalDigits)"

            result = result.replacingCharacters(in: swiftRange, with: "\(transformed)")
        }

        return result
    }

    private static let currencies: [Character: (bill: String, cent: String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent"),
    ]

    private static let currencyRegex = try! NSRegularExpression(
        pattern: #"[\$£€]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[\$£€]\d+\.\d\d?\b"#
    )

    private static let timeRegex = try! NSRegularExpression(
        pattern: #"\b(?:[1-9]|1[0-2]):[0-5]\d\b"#
    )

    private static let decimalRegex = try! NSRegularExpression(
        pattern: #"\b\d*\.\d+\b"#
    )

    private static let rangeRegex = try! NSRegularExpression(
        pattern: #"([\$£€]?\d+)-([\$£€]?\d+)"#
    )

    private static let commaInNumberRegex = try! NSRegularExpression(
        pattern: #"(^|[^\d])(\d+(?:,\d+)*)([^\d]|$)"#
    )

    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#
    )
}
