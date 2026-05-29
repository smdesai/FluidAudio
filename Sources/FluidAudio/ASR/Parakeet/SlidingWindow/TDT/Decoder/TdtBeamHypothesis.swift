@preconcurrency import CoreML
import Foundation

// MARK: - Beam configuration

/// Configuration for `TdtBeamDecoder`.
///
/// All defaults reproduce greedy decoding:
/// - `beamSize=1` is greedy (single hypothesis), matching `TdtDecoderV3`.
/// - Bias config is nil (no shallow fusion), so the beam just argmaxes.
public struct TdtBeamConfig: Sendable {

    /// Number of hypotheses kept per step. `1` = greedy.
    public var beamSize: Int = 4

    /// Per-token log-prob below which expansions are ignored (only top-K
    /// expansions per parent are considered, where K = beamSize).
    public var topKPerHypothesis: Int = 4

    /// Length normalization exponent used when ranking final hypotheses.
    /// Final score is `logProb / max(1, tokenCount)^lengthPenalty`.
    /// 1.0 = full normalization; 0.0 = raw log-prob (favors short).
    public var lengthPenalty: Float = 1.0

    /// Hypotheses with `logProb < bestLogProb - pruningThreshold` are
    /// dropped from the beam. Set very high to disable.
    public var pruningThreshold: Float = 5.0

    /// When the joint's argmax token is blank and the second-best token's
    /// log-prob is at least this many points below blank, skip top-K
    /// expansion entirely and emit only blank. TDT is blank-dominant, so
    /// most frames qualify and the savings (no state copy, no top-K, no
    /// duplicate hypotheses to prune) are large. Set to `+infinity` to
    /// disable and always do top-K.
    ///
    /// Default 6.0 log-prob units (~e^6 ≈ 400× ratio between blank and
    /// the runner-up). At this margin, the second-best token's
    /// contribution to the beam is dominated by other paths' main
    /// emissions, so dropping it costs nothing in practice.
    public var blankShortcutMargin: Float = 6.0

    /// Maximum non-blank tokens emitted at the same encoder frame before
    /// a hypothesis is forced to advance. Mirrors `TdtConfig.maxSymbolsPerStep`.
    public var maxSymbolsPerStep: Int = 5

    /// Optional shallow-fusion biasing of vocabulary keywords during decode.
    public var bias: TdtBeamBiasConfig? = nil

    public init(
        beamSize: Int = 4,
        topKPerHypothesis: Int = 4,
        lengthPenalty: Float = 1.0,
        pruningThreshold: Float = 5.0,
        blankShortcutMargin: Float = 6.0,
        maxSymbolsPerStep: Int = 5,
        bias: TdtBeamBiasConfig? = nil
    ) {
        self.beamSize = max(1, beamSize)
        self.topKPerHypothesis = max(1, topKPerHypothesis)
        self.lengthPenalty = lengthPenalty
        self.pruningThreshold = pruningThreshold
        self.blankShortcutMargin = blankShortcutMargin
        self.maxSymbolsPerStep = max(1, maxSymbolsPerStep)
        self.bias = bias
    }
}

/// Shallow-fusion configuration for keyword biasing inside beam decode.
///
/// At each step, hypotheses that have started a partial keyword match
/// (their previously emitted tokens are a prefix of some vocab term's
/// token sequence) get a `bonus` log-prob added to the next token of
/// that term. This pulls the beam toward completing a known keyword
/// when the audio supports it, without forcing it.
public struct TdtBeamBiasConfig: Sendable {

    /// Tokenized vocabulary (ID sequences in TDT vocab space). Each inner
    /// array is the token sequence for one keyword (typically 2-6 tokens).
    public let keywordTokenSequences: [[Int]]

    /// Log-prob bonus added to the next-token logit when a hypothesis is
    /// mid-match for a keyword. Same semantic as `cbw` in the existing
    /// rescorer; default 4.5 mirrors `ContextBiasingConstants.defaultCbw`
    /// after adaptive scaling.
    public var bonus: Float = 4.5

    /// Optional detection windows from the CTC context graph. When present,
    /// keyword first-token activation is allowed only inside matching windows.
    /// Continuation tokens remain biased once a phrase has started.
    public let windows: [TdtBeamBiasWindow]

    public init(keywordTokenSequences: [[Int]], bonus: Float = 4.5, windows: [TdtBeamBiasWindow] = []) {
        self.keywordTokenSequences = keywordTokenSequences.filter { !$0.isEmpty }
        self.bonus = bonus
        self.windows = windows
    }
}

public struct TdtBeamBiasWindow: Sendable {
    public let keywordIndex: Int
    public let startFrame: Int
    public let endFrame: Int

    public init(keywordIndex: Int, startFrame: Int, endFrame: Int) {
        self.keywordIndex = keywordIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
}

// MARK: - Beam hypothesis

/// State tracked per active partial keyword match. A hypothesis can have
/// multiple of these simultaneously (e.g. when a token starts several
/// keywords) — they're tracked independently and dropped when they fail
/// to advance.
struct TdtBeamBiasMatch: Sendable {
    /// Index into `keywordTokenSequences`.
    let keywordIndex: Int
    /// Number of tokens of that keyword the hypothesis has already
    /// emitted. The next-token bonus targets `keyword[position]`.
    var position: Int
}

/// One hypothesis in the beam. Owns its decoder state and the cached
/// joint-side decoder projection so siblings don't clobber each other.
struct TdtBeamHypothesis {
    var tokens: [Int]
    var timestamps: [Int]
    var tokenConfidences: [Float]
    var tokenDurations: [Int]
    var logProb: Float
    var lastToken: Int?

    /// Encoder-frame index where the next decision will be scored. The
    /// beam advances each hypothesis independently because TDT can skip
    /// 0-4 frames per emission.
    var timeIndex: Int

    /// LSTM state (h, c) at the current step.
    var state: TdtDecoderState

    /// Most recent decoder projection (joint input). Reused while emitting
    /// blanks (decoder LSTM only advances on non-blank emissions, matching
    /// `TdtDecoderV3`).
    var lastDecoderProjection: MLMultiArray?

    /// Active partial keyword matches (only used when bias config is set).
    var biasMatches: [TdtBeamBiasMatch]

    /// Bias windows consumed by this hypothesis. A CTC detection should seed
    /// a phrase at most once per hypothesis to avoid repeated phrase loops.
    var consumedBiasWindows: Set<Int>

    /// Tokens emitted at the current `timeIndex`. Used to enforce
    /// `maxSymbolsPerStep` so a hypothesis can't get stuck.
    var symbolsAtCurrentFrame: Int

    /// Score normalized for ranking against other hypotheses. Length
    /// penalty avoids a constant bias toward short hypotheses (which
    /// accumulate fewer log-probs).
    func normalizedScore(lengthPenalty: Float) -> Float {
        guard !tokens.isEmpty else { return logProb }
        let n = Float(tokens.count)
        if lengthPenalty == 0 { return logProb }
        return logProb / powf(n, lengthPenalty)
    }

    /// Convert to the hypothesis type used by the rest of the pipeline.
    func asTdtHypothesis() -> TdtHypothesis {
        var h = TdtHypothesis(decState: state)
        h.score = logProb
        h.ySequence = tokens
        h.timestamps = timestamps
        h.tokenConfidences = tokenConfidences
        h.tokenDurations = tokenDurations
        h.lastToken = lastToken
        return h
    }
}
