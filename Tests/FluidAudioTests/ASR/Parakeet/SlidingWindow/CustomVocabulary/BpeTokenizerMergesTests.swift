import XCTest

@testable import FluidAudio

/// Tests that `BpeTokenizer` accepts both `merges` serializations emitted by
/// the HuggingFace `tokenizers` library:
///   - legacy string form:  `["a b", "c d", ...]`   (tokenizers < 0.20)
///   - array form:          `[["a","b"], ["c","d"]]` (tokenizers >= 0.20)
///
/// These build a minimal in-memory `tokenizer.json` on disk so the merge
/// parsing branch is exercised without requiring a downloaded model.
final class BpeTokenizerMergesTests: XCTestCase {

    /// Minimal BPE tokenizer JSON. `mergesJSON` is spliced in verbatim so each
    /// test can supply either serialization. The vocab contains the single
    /// characters plus the merged token so a successful merge is observable.
    private func writeTokenizer(mergesJSON: String) throws -> URL {
        let json = """
            {
              "model": {
                "type": "BPE",
                "vocab": { "a": 0, "b": 1, "ab": 2 },
                "merges": \(mergesJSON)
              },
              "added_tokens": [ { "id": 3, "content": "<unk>" } ]
            }
            """
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bpe-merges-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try json.write(
            to: dir.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        return dir
    }

    func testLegacyStringMergesAreApplied() throws {
        let dir = try writeTokenizer(mergesJSON: #"["a b"]"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try BpeTokenizer.load(from: dir)
        // chars ["a","b"] merge via the "a b" rule into the single "ab" token.
        XCTAssertEqual(tok.encode("ab", prependWordBoundary: false), [2])
    }

    func testArrayFormMergesAreApplied() throws {
        let dir = try writeTokenizer(mergesJSON: #"[["a","b"]]"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let tok = try BpeTokenizer.load(from: dir)
        // Array-form merges must produce the identical result to the legacy form.
        XCTAssertEqual(tok.encode("ab", prependWordBoundary: false), [2])
    }

    func testBothMergeFormsProduceIdenticalTokenization() throws {
        let legacyDir = try writeTokenizer(mergesJSON: #"["a b"]"#)
        let arrayDir = try writeTokenizer(mergesJSON: #"[["a","b"]]"#)
        defer {
            try? FileManager.default.removeItem(at: legacyDir)
            try? FileManager.default.removeItem(at: arrayDir)
        }

        let legacy = try BpeTokenizer.load(from: legacyDir)
        let array = try BpeTokenizer.load(from: arrayDir)
        XCTAssertEqual(
            legacy.encode("ab", prependWordBoundary: false),
            array.encode("ab", prependWordBoundary: false))
    }
}
