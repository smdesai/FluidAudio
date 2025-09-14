#!/usr/bin/env python3
"""
main5.py — zero‑config batch runner for the Swift TTS CLI.

Generates a list of texts and synthesizes each one via:
    swift run fluidaudio tts "<text>"

The first run includes --auto-download to ensure models are present.
Outputs are saved under ./tts_outputs/out_XX.wav with a manifest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from collections import OrderedDict


def ensure_swift() -> None:
    try:
        subprocess.run(["swift", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("swift is not available on PATH. Install Xcode Command Line Tools.", file=sys.stderr)
        sys.exit(2)

    # Prebuild once to avoid repeated rebuilds during swift run invocations
    try:
        subprocess.run(["swift", "build"], check=True)
    except Exception:
        # Non-fatal; continue and let `swift run` build as needed
        pass


def resolve_cli() -> list[str]:
    """Return argv prefix to invoke the CLI efficiently.
    Prefer the built binary (avoids repeated `swift run` build noise),
    fallback to `swift run fluidaudio`.
    """
    debug_bin = Path(".build/debug/fluidaudio")
    release_bin = Path(".build/release/fluidaudio")
    if debug_bin.exists():
        return [str(debug_bin)]
    if release_bin.exists():
        return [str(release_bin)]
    return ["swift", "run", "fluidaudio"]


def texts_to_synthesize() -> list[str]:
    # Curated set covering various prosody and punctuation cases
    return [
        "Hello there.",
        "How are you today?",
        "FluidAudio makes text to speech fast, accurate, and fun.",
        "Short.",
        "This is a longer sentence that should still fit within a single chunk for most token budgets, but pushes punctuation handling and timing across clauses.",
        "Wait—what just happened?",
        "Well... that was unexpected.",
        "Mr. Smith went to Washington. He arrived at 10:30 a.m.",
        "Read, then pause; continue: now finish.",
        "Parentheses (with content) should not break flow.",
        "Quotes: “Hello,” she said. “Goodbye,” he replied.",
        "Two sentences. Then a paragraph break follows.",
        "New paragraph starts here, to test paragraph-level pause.",
        "A very long sentence without many commas that forces chunking by token limits because it keeps adding words and stretches the capacity of the model input which is useful to ensure the chunker splits before overrunning the target tokens and stitches results smoothly at boundaries.",
        "Edge-cases: URLs like https://example.com and e-mail test@example.com.",
        "Okay, last one: let’s ensure emojis don’t crash anything.",
    ]

def run_batch() -> int:
    texts = texts_to_synthesize()
    out_dir = Path("tts_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.txt"

    # Truncate manifest for a clean run
    if manifest_path.exists():
        manifest_path.unlink()

    # Synthesize each text
    cli = resolve_cli()
    for idx, text in enumerate(texts, start=1):
        outfile = out_dir / f"out_{idx:02d}.wav"
        cmd = [*cli, "tts", text]
        if idx == 1:
            cmd.append("--auto-download")
        cmd += ["--output", str(outfile)]
        print(f"[{idx}/{len(texts)}] $", " ".join([c if c else '""' for c in cmd]))
        try:
            # Capture to reduce console noise if running via built binary; otherwise show swift run logs
            if cmd[0].endswith("fluidaudio"):
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed for index {idx} with exit code {e.returncode}")
            return e.returncode

        # Append to manifest (fresh for each run)
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(f"{outfile}\t{text}\n")

    print(f"Wrote {len(texts)} files to {out_dir} and manifest to {manifest_path}")

    # Run ASR on each generated file and capture hypotheses
    asr_tsv = out_dir / "asr_hyp.tsv"
    # Deduplicate manifest lines by audio path to avoid repeated ASR calls
    unique_items: "OrderedDict[str, str]" = OrderedDict()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                path_str, ref = line.split("\t", 1)
            except ValueError:
                continue
            unique_items[path_str] = ref

    with open(asr_tsv, "w", encoding="utf-8") as out:
        for i, (path_str, ref) in enumerate(unique_items.items(), start=1):
            cmd = [*cli, "transcribe", path_str]
            print(f"[{i}/{len(unique_items)}] $", " ".join(cmd))
            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                hyp = extract_final_transcription(res.stdout)
                # Retry with streaming mode if batch failed or produced empty hyp
                if not hyp or hyp.startswith("Batch transcription failed:"):
                    cmd_stream = [*cli, "transcribe", path_str, "--streaming"]
                    print(f"    retry streaming -> $", " ".join(cmd_stream))
                    try:
                        res2 = subprocess.run(cmd_stream, check=True, capture_output=True, text=True)
                        hyp2 = extract_final_transcription(res2.stdout)
                        if hyp2:
                            hyp = hyp2
                    except subprocess.CalledProcessError:
                        pass
            except subprocess.CalledProcessError as e:
                hyp = ""
                print(f"ASR failed for {path_str}: exit {e.returncode}")
            out.write(f"{path_str}\t{ref}\t{hyp}\n")

    print(f"ASR results written to {asr_tsv}")
    # Print comparisons and simple WER summary
    print("\nASR Comparisons (raw)\n=====================")
    overall_edits = 0
    overall_ref_words = 0
    try:
        rows = asr_tsv.read_text(encoding="utf-8").splitlines()
    except Exception:
        rows = []
    for idx, line in enumerate(rows, start=1):
        try:
            path_str, ref, hyp = line.split("\t", 2)
        except ValueError:
            continue
        name = Path(path_str).name
        wer_val, edits, ref_len = wer_light(ref, hyp)
        overall_edits += edits
        overall_ref_words += ref_len
        print(f"[{idx:02d}] {name}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print(f"  WER(light): {wer_val:.3f}\n")
    if overall_ref_words > 0:
        overall = overall_edits / overall_ref_words
        print(f"Overall WER(light): {overall:.3f}  (edits={overall_edits}, ref_words={overall_ref_words})")
    return 0


def extract_final_transcription(stdout: str) -> str:
    """Parse TranscribeCommand output to get the first final transcription line."""
    lines = stdout.splitlines()
    # If batch failed, surface empty to trigger streaming retry upstream
    for l in lines:
        if l.strip().startswith("Batch transcription failed:"):
            return ""
    for i, l in enumerate(lines):
        if l.strip().startswith("Final transcription:"):
            # Collect next non-empty line(s) until a separator or section; take first line as hypothesis
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if not s:
                    continue
                # Stop if a separator or header appears
                if set(s) <= {"=", "-"} and len(s) > 5:
                    break
                return s
            break
    # Fallback: try last non-empty line
    for l in reversed(lines):
        if l.strip():
            return l.strip()
    return ""


def wer_light(ref: str, hyp: str) -> tuple[float, int, int]:
    """WER with light normalization (lowercase, basic Unicode punctuation normalization)."""
    def norm(x: str) -> str:
        t = x.lower()
        t = (
            t.replace("\u2019", "'")
             .replace("\u2018", "'")
             .replace("\u201c", '"')
             .replace("\u201d", '"')
             .replace("\u2014", "-")
             .replace("\u2026", "...")
        )
        return " ".join(t.split())

    def tokenize(x: str) -> list[str]:
        out: list[str] = []
        cur = []
        for ch in x:
            if ch.isalnum() or ch == "'":
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    r = tokenize(norm(ref))
    h = tokenize(norm(hyp))
    n, m = len(r), len(h)
    if n == 0:
        return (0.0 if m == 0 else 1.0, m, n)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    edits = dp[n][m]
    return (edits / n, edits, n)


def main() -> int:
    ensure_swift()
    return run_batch()


if __name__ == "__main__":
    raise SystemExit(main())
