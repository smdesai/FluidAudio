# Vocabulary Rescoring vs Keyword Boosting

Two distinct systems for domain-specific term recognition in FluidAudio ASR, operating at different stages of the pipeline.

## 1. Vocabulary Rescoring (`--custom-vocab`) — Post-Decode

The TDT decoder runs normally (greedy decoding), produces a transcript, then a **second model** (Parakeet-CTC) re-examines the audio to find and fix misrecognized terms.

### Pipeline

```
Audio → TDT Encoder → TDT Decoder (greedy) → Transcript
                                                  ↓
Audio → CTC Model → Log Probabilities → VocabularyRescorer
                                                  ↓
                                          Corrected Transcript
```

### Key Properties

- Uses a **separate CTC model** (`parakeet-ctc-0.6b-coreml`) for acoustic verification
- Works on the **finished text** — finds words that look/sound similar to vocab terms and replaces them
- String similarity matching (how close is "cab drivers" to "cab driver"?) + acoustic score from CTC log probabilities
- Can correct words the TDT decoder got completely wrong (e.g., "pie torch" → "PyTorch")
- **Doubles inference cost** — runs both TDT and CTC models on the same audio
- Configured via `configureVocabularyBoosting()` / CLI `--custom-vocab <file>`

### Code Path

- `AsrManager.configureVocabularyBoosting()` — sets up CTC spotter + rescorer
- `AsrTranscription.applyVocabularyRescoring()` — runs CTC inference and replaces words
- `VocabularyRescorer.ctcTokenRescore()` — core rescoring logic
- `CtcKeywordSpotter` — runs CTC model to get log probabilities

## 2. Keyword Boosting (`--phrases`) — During Decode

While the TDT decoder is choosing tokens frame-by-frame, a prefix trie checks if any top-k candidate continues a keyword path. If so, the keyword token's logit gets a boost, biasing the decoder toward emitting the keyword.

### Pipeline

```
Audio → TDT Encoder → TDT Decoder ←→ KeywordPrefixTrie (biasing at each step)
                          ↓
                     Transcript (already contains boosted terms)
```

### Key Properties

- Operates **inside the decoder loop** — no second model needed
- Requires `JointDecisionv2.mlmodelc` which outputs top-k token IDs + logits (not just the greedy best)
- Only boosts tokens that are **already acoustically plausible** (must appear in top-64 candidates)
- Cannot correct a word the model never considered — it can only promote a likely-but-not-top-1 token to top-1
- **Zero additional inference cost** — same model, same forward pass
- Configured via `configureKeywordBoosting()` / CLI `--phrases "term1,term2"`

### Code Path

- `AsrManager.configureKeywordBoosting()` — tokenizes terms, builds prefix trie
- `TdtGreedyTokenizer.encodeAllPaths()` — enumerates all valid BPE decompositions (up to 64 paths per term)
- `KeywordPrefixTrie` + `TrieCursor` — tracks token-by-token match state during decoding
- `TdtDecoderV3.applyKeywordBiasing()` — checks top-k candidates against trie, boosts matching tokens
- `DetectedPhrase` — records matched phrases with timing, confidence, and whether boosting was applied
- `KeywordBoostingContext` — holds trie, boost weight, callback, and accumulated detections

## Comparison

| | Vocabulary Rescoring | Keyword Boosting |
|---|---|---|
| **When** | After transcription | During decoding |
| **Model** | Separate CTC model | Same JointDecisionv2 |
| **Cost** | 2x inference | ~0 additional |
| **Correction power** | Can fix any word | Only if in top-64 |
| **Latency** | Higher (second pass) | None (inline) |
| **CLI flag** | `--custom-vocab file` | `--phrases "a,b"` |
| **Input format** | File (one term per line) | Inline comma-separated |
| **Detection output** | Replacement log | `DetectedPhrase` with timing |
| **Streaming support** | Hybrid rescoring on confirmed chunks | Real-time biasing per frame |

## Using Both Together

Currently they are independent code paths that don't interact. If both `--phrases` and `--custom-vocab` are provided, both run:

1. **Keyword boosting runs first** (during decode) — nudges the decoder toward correct tokens when acoustically plausible
2. **Vocabulary rescoring runs second** (post-decode) — catches anything boosting missed because the term wasn't in top-64

The rescoring pass does not know which words were already boosted, so it may redundantly rescore words that were already correct. A natural improvement would be to skip rescoring on words that keyword boosting already detected (using the `detectedPhrases` list to mark regions as "already handled").

## CLI Examples

```bash
# Keyword boosting only (lightweight, real-time)
fluidaudiocli transcribe audio.wav --phrases "NVIDIA,PyTorch"

# Vocabulary rescoring only (heavier, more powerful)
fluidaudiocli transcribe audio.wav --custom-vocab vocab.txt

# Both together
fluidaudiocli transcribe audio.wav --phrases "NVIDIA,PyTorch" --custom-vocab vocab.txt

# Streaming with keyword boosting
fluidaudiocli transcribe audio.wav --phrases "cab driver" --stream

# Adjust boost strength (default: 3.0)
fluidaudiocli transcribe audio.wav --phrases "cab driver" --boost-weight 5.0
```
