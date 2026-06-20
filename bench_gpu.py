"""Turnkey GPU benchmark for the Hinglish path (run on the Linux+GPU target box).

Measures, per backend, the metrics the hackathon cares about:
    load time · p50 · p95 · throughput (clips/s) · peak VRAM · WER · Meaning

It drives the SAME qwen-asr model already wired in solution/transcribe.py, so it
benchmarks exactly what ships. CPU-only boxes will report that no CUDA is present.

    python bench_gpu.py                      # auto: vLLM if installed, else transformers
    STT_QWEN_BACKEND=vllm        python bench_gpu.py
    STT_QWEN_BACKEND=transformers python bench_gpu.py
    python bench_gpu.py --warmup 1 --runs 3  # warmup + repeats for stable p50/p95

Notes:
  * Faithfulness / Latin-tech-term preservation are reported from the FULL-token diff
    (Devanagari kept; not the ASCII-only scorecard proxy).
  * For AWQ/GPTQ: no CUDA-loadable quantized checkpoint of this fine-tune exists; this
    script does NOT self-quantize (quality risk on a 0.6B custom arch). See the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scorecard import judge_meaning, percentile, wer  # noqa: E402
import solution.transcribe as T  # noqa: E402

_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_TECH = ["impress", "document", "formatting", "tutorial", "workspace", "slide", "font"]


def _faithful(gold: str, pred: str) -> dict:
    """Code-switch faithfulness signals (not the ASCII proxy)."""
    return {
        "kept_devanagari": bool(_DEVA.search(pred)),
        "kept_latin": bool(_LATIN.search(pred)),
        "latin_terms_kept": sum(1 for t in _TECH if t in gold and t in pred),
        "latin_terms_expected": sum(1 for t in _TECH if t in gold),
    }


def _vram_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2, help="repeats per clip for p50/p95")
    args = ap.parse_args()

    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else "(none)"
    except Exception:
        cuda, gpu_name = False, "(torch missing)"
    backend_pref = os.environ.get("STT_QWEN_BACKEND", "auto")
    print(f"CUDA={cuda}  GPU={gpu_name}  STT_QWEN_BACKEND={backend_pref}")
    if not cuda:
        print("\nNO GPU DETECTED — this benchmark must run on the Linux+GPU target box.")
        print("On CPU the Hinglish path is ~60s p50 / ~160s p95 (already measured).")
        return

    man = {c["clip_id"]: c for c in json.load(open(os.path.join(HERE, "samples/manifest.json"), encoding="utf-8"))}
    hin = [cid for cid in man if (man[cid].get("language") or "").lower() in ("hi-en", "hinglish")]
    clips = [(os.path.join(HERE, "samples", cid + ".wav"), man[cid]["gold"]) for cid in hin]

    t0 = time.time()
    h = T.get_hinglish_model()
    load_s = time.time() - t0
    if h is None:
        print("Hinglish model failed to load:", T._LAST_HINGLISH_ERROR)
        return
    meta = dict(T._HINGLISH_META)
    print(f"\nLOADED  backend={meta.get('backend')}  precision={meta.get('precision')}  "
          f"load={load_s:.1f}s  footprint={meta.get('footprint_mb')}MB")

    audios = [(T.load_audio(w), g) for w, g in clips]
    for a, _ in audios[:args.warmup]:          # warmup (CUDA graphs / caches)
        T.hinglish_transcribe(a)

    lat, wers, means, faith = [], [], [], []
    t_start = time.time()
    for audio, gold in audios:
        for _ in range(args.runs):
            s = time.time()
            r = T.hinglish_transcribe(audio)
            lat.append((time.time() - s) * 1000)
            txt = r["text"]
            wers.append(wer(gold, txt)); means.append(judge_meaning(gold, txt))
            faith.append(_faithful(gold, txt))
    wall = time.time() - t_start
    n = len(lat)
    lt = sum(f["latin_terms_kept"] for f in faith)
    le = sum(f["latin_terms_expected"] for f in faith) or 1

    print("\n=== HINGLISH GPU BENCHMARK ===")
    print(f"  backend        : {meta.get('backend')} ({meta.get('precision')})")
    print(f"  load time      : {load_s:.1f} s")
    print(f"  p50 latency    : {percentile(lat, .50):.0f} ms")
    print(f"  p95 latency    : {percentile(lat, .95):.0f} ms")
    print(f"  throughput     : {n / wall:.2f} clips/s")
    print(f"  peak VRAM      : {_vram_mb():.0f} MB" if _vram_mb() else "  peak VRAM      : n/a")
    print(f"  WER (proxy)    : {sum(wers)/n:.3f}   (ASCII-only; directional)")
    print(f"  Meaning (proxy): {sum(means)/n:.3f}")
    print(f"  faithfulness   : devanagari_kept={all(f['kept_devanagari'] for f in faith)}  "
          f"latin_terms_kept={lt}/{le}")
    print("\n(Quality on hidden set is judged by an LLM, not this proxy — confirm transcripts read faithfully.)")


if __name__ == "__main__":
    main()
