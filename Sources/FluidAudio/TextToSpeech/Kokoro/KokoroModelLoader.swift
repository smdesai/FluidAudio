import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
actor KokoroModelLoader {
    static let shared = KokoroModelLoader()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModelLoader")
    private var kokoroModels: [ModelNames.TTS.Variant: MLModel] = [:]
    private var tokenLengthCache: [ModelNames.TTS.Variant: Int] = [:]
    private var downloadedModelBundle: TtsModels?
    private var referenceDimension: Int?


    private let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"

    private func downloadFileIfNeeded(filename: String, urlPath: String) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)
        guard !FileManager.default.fileExists(atPath: localURL.path) else {
            logger.info("File already exists: \(filename)")
            return
        }

        logger.info("Downloading \(filename)...")
        guard let downloadURL = URL(string: "\(baseURL)/\(urlPath)") else {
            throw TTSError.modelNotFound("Invalid URL for \(filename)")
        }

        let (data, response) = try await DownloadUtils.sharedSession.data(from: downloadURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw TTSError.modelNotFound("Failed to download \(filename)")
        }

        try data.write(to: localURL)
        logger.info("Downloaded \(filename) (\(data.count) bytes)")
    }

    func ensureRequiredFiles() async throws {
        try await downloadFileIfNeeded(filename: "us_gold.json", urlPath: "us_gold.json")
        try await downloadFileIfNeeded(filename: "us_silver.json", urlPath: "us_silver.json")

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelsDirectory = cacheDir.appendingPathComponent("Models")
        _ = try? await DownloadUtils.ensureEspeakDataBundle(in: modelsDirectory)
    }

    func loadModelsIfNeeded() async throws {
        if kokoroModels.count == ModelNames.TTS.Variant.allCases.count {
            return
        }

        if downloadedModelBundle == nil {
            downloadedModelBundle = try await TtsModels.download()
        }

        guard let bundle = downloadedModelBundle else {
            throw TTSError.modelNotFound("Kokoro models unavailable after download attempt")
        }

        for variant in ModelNames.TTS.Variant.allCases where kokoroModels[variant] == nil {
            guard let model = bundle.model(for: variant) else {
                throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
            }
            kokoroModels[variant] = model
            tokenLengthCache[variant] = KokoroModel.inferTokenLength(from: model)
            logger.info("Loaded Kokoro \(variantDescription(variant)) model from cache")
        }

        let loadedVariants = kokoroModels.keys.map { variantDescription($0) }.sorted().joined(separator: ", ")
        logger.info("Kokoro models ready: [\(loadedVariants)]")
    }

    func model(for variant: ModelNames.TTS.Variant) async throws -> MLModel {
        if let existing = kokoroModels[variant] {
            return existing
        }
        try await loadModelsIfNeeded()
        guard let model = kokoroModels[variant] else {
            throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
        }
        return model
    }

    func tokenLength(for variant: ModelNames.TTS.Variant) async throws -> Int {
        if let cached = tokenLengthCache[variant] {
            return cached
        }
        let model = try await model(for: variant)
        let length = KokoroModel.inferTokenLength(from: model)
        tokenLengthCache[variant] = length
        return length
    }

    func referenceEmbeddingDimension() async throws -> Int {
        if let cached = referenceDimension { return cached }
        let model = try await model(for: ModelNames.TTS.defaultVariant)
        let dim = KokoroModel.refDim(from: model)
        referenceDimension = dim
        return dim
    }

    func registerPreloadedModels(_ models: TtsModels) {
        downloadedModelBundle = models

        for variant in ModelNames.TTS.Variant.allCases {
            if let model = models.model(for: variant) {
                kokoroModels[variant] = model
                tokenLengthCache[variant] = KokoroModel.inferTokenLength(from: model)
            }
        }
        if referenceDimension == nil,
            let defaultModel = models.model(for: ModelNames.TTS.defaultVariant) {
            referenceDimension = KokoroModel.refDim(from: defaultModel)
        }
    }

    private func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }
}
