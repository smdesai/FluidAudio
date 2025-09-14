import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    public static func run(arguments: [String]) async {
        // Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--auto-download]
        guard arguments.count >= 1 else {
            print("Usage: fluidaudio tts \"text\" [--output file.wav] [--voice af_heart] [--auto-download]")
            return
        }

        let text = arguments[0]
        var output = "output.wav"
        var voice = "af_heart"
        var autoDownload = false

        for i in 0..<arguments.count {
            if arguments[i] == "--output" || arguments[i] == "-o", i + 1 < arguments.count { output = arguments[i + 1] }
            if arguments[i] == "--voice" || arguments[i] == "-v", i + 1 < arguments.count { voice = arguments[i + 1] }
            if arguments[i] == "--auto-download" { autoDownload = true }
        }

        do {
            if autoDownload {
                try await KokoroModel.ensureRequiredFiles()
            }
            // Use full KokoroModel pipeline (with chunking + punctuation-driven pauses)
            let wav = try await KokoroModel.synthesize(text: text, voice: voice)
            let outURL = URL(fileURLWithPath: output)
            try wav.write(to: outURL)
            print("Saved: \(outURL.path)")
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
}
