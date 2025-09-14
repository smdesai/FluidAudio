#if canImport(CEspeakNG)
import CEspeakNG
import Foundation
import OSLog

/// Thread-safe wrapper around eSpeak NG C API to get IPA phonemes for a word.
/// Uses espeak_TextToPhonemes with IPA mode.
@available(macOS 13.0, iOS 16.0, *)
final class EspeakG2P {
    static let shared = EspeakG2P()
    private let queue = DispatchQueue(label: "com.fluidaudio.tts.espeak.g2p")
    private var initialized = false

    private init() {
        _ = initializeIfNeeded()
    }

    private func initializeIfNeeded() -> Bool {
        if initialized { return true }
        let rc = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nil, 0)
        guard rc >= 0 else { return false }
        _ = "en-us".withCString { espeak_SetVoiceByName($0) }
        initialized = true
        return true
    }

    /// Return IPA tokens for a word, or nil on failure.
    func phonemize(word: String) -> [String]? {
        return queue.sync {
            guard initializeIfNeeded() else { return nil }
            return word.withCString { cstr -> [String]? in
                var raw: UnsafeRawPointer? = UnsafeRawPointer(cstr)
                let modeIPA = Int32(espeakPHONEMES_IPA)
                let textmode = Int32(espeakCHARS_AUTO)
                guard let outPtr = espeak_TextToPhonemes(&raw, textmode, modeIPA) else { return nil }
                let s = String(cString: outPtr)
                if s.isEmpty { return nil }
                if s.contains(where: { $0.isWhitespace }) {
                    return s.split{ $0.isWhitespace }.map { String($0) }
                } else {
                    return s.unicodeScalars.map { String($0) }
                }
            }
        }
    }
}
#endif
