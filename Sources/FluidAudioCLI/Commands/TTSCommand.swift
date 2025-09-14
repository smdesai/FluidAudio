import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    public static func run(arguments: [String]) async {
        // Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--auto-download]
        guard !arguments.isEmpty else {
            printUsage()
            return
        }

        let text = arguments[0]
        var output = "output.wav"
        var voice = "af_heart"
        // Always ensure required files in CLI

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                return
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--voice", "-v":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                // No-op: downloads are always ensured by the CLI
                ()
            default:
                // Unknown flag; ignore for forward-compat
                ()
            }
            i += 1
        }

        do {
            // Always ensure required files (model + dictionary)
            try await KokoroModel.ensureRequiredFiles()
            // Use full KokoroModel pipeline (with chunking + punctuation-driven pauses)
            let wav = try await KokoroModel.synthesize(text: text, voice: voice)
            let outURL = URL(fileURLWithPath: output)
            try wav.write(to: outURL)
            print("Saved: \(outURL.path)")
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart]

            Options:
              --output, -o         Output WAV path (default: output.wav)
              --voice, -v          Voice name (default: af_heart)
              (models/dictionary auto-download is always on in CLI)
              --help, -h           Show this help
            """
        )
    }
}
