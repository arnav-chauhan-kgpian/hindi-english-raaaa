"""bench_stream.py — local latency proxy for the streaming/dictation track.

Mimics the sealed stream_server.py: feeds each clip's PCM to solution.draft.draft() as
20 ms frames, then measures the three scored latency metrics. Run it on the SAME class of
hardware as the scoring box (Apple-silicon M1; the MPS Qwen path). On a GPU-less box the
numbers are not representative — Qwen runs on CPU and will be much slower than the M1.

    python bench_stream.py                       # all samples/*.wav, 1 run each
    python bench_stream.py --runs 3 a.wav b.wav  # median of 3 runs per clip

Reports, per clip and as p50/p95 across clips:
  * end_to_final  — wall time of the draft(full_buffer, is_final=True) call  (target ≤2000ms)
  * ttfs          — wall time until the first committed partial (stable_chars>0) (target ≤1000ms)
  * churn         — committed characters that were later rewritten (target ~0)

NOTE: end_to_final is the metric the score hinges on (25 pts). ttfs here is compute-bound
(frames fed back-to-back); under real-time arrival the real TTFS is max(this, audio elapsed).
The official harness is authoritative — this is a fast proxy to decide if the synchronous
baseline already clears ≤2000ms before adding a speculative-final optimization.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import time
import wave

from solution import draft as D
from solution import transcribe as T

FRAME_MS = 20
SR = 16000
FRAME_BYTES = int(SR * FRAME_MS / 1000) * 2  # s16le → 2 bytes/sample


def _device() -> str:
    if T._cuda_available():
        return "CUDA"
    if T._mps_available():
        return "Apple MPS (representative of the M1 scoring box)"
    return "CPU ONLY — NOT representative of the M1; Qwen will be far slower here"


def _read_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError(f"{path}: need 16 kHz mono, got {w.getframerate()} Hz "
                             f"{w.getnchannels()}ch")
        return w.readframes(w.getnframes())


def _bench_once(pcm: bytes) -> dict:
    """One streaming pass over a clip. Returns the three latency metrics (ms)."""
    committed = ""           # the prefix we've promised to keep
    churn = 0                # committed chars that later changed
    ttfs_ms = None
    t_start = time.perf_counter()

    # feed cumulative 20 ms frames (the contract's wire cadence)
    for end in range(FRAME_BYTES, len(pcm) + 1, FRAME_BYTES):
        text, stable = D.draft(pcm[:end], False)
        new_commit = text[:stable]
        # churn = chars where the new committed prefix disagrees with the old one
        for a, b in zip(committed, new_commit):
            if a != b:
                churn += 1
        if len(committed) > len(new_commit):
            churn += len(committed) - len(new_commit)   # un-committed = also churn
        committed = new_commit
        if ttfs_ms is None and stable > 0:
            ttfs_ms = (time.perf_counter() - t_start) * 1000

    # the moment the user stops → time JUST the final call (end-to-final)
    t_final = time.perf_counter()
    final_text, _ = D.draft(pcm, True)
    end_to_final_ms = (time.perf_counter() - t_final) * 1000

    return {
        "end_to_final_ms": end_to_final_ms,
        "ttfs_ms": ttfs_ms if ttfs_ms is not None else float("nan"),
        "churn": churn,
        "final_len": len(final_text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", help="wav paths (default: samples/*.wav)")
    ap.add_argument("--runs", type=int, default=1, help="runs per clip; report the median")
    args = ap.parse_args()

    clips = args.clips or sorted(glob.glob("samples/*.wav"))
    if not clips:
        print("no clips found (pass wav paths or put 16k mono wavs in samples/)")
        return

    print(f"device: {_device()}")
    print("warming up (loading models + a dry decode)…")
    D.warmup()
    try:                       # one full dry pass so the very first timed clip is hot
        _bench_once(_read_pcm(clips[0]))
    except Exception as e:
        print(f"  warmup pass skipped: {type(e).__name__}: {e}")

    rows = []
    print(f"\n{'clip':40s} {'end→final':>11s} {'ttfs':>9s} {'churn':>6s}")
    print("-" * 70)
    for path in clips:
        try:
            pcm = _read_pcm(path)
        except Exception as e:
            print(f"{path[:40]:40s}  SKIP: {e}")
            continue
        runs = [_bench_once(pcm) for _ in range(max(1, args.runs))]
        e2f = statistics.median(r["end_to_final_ms"] for r in runs)
        ttfs = statistics.median(r["ttfs_ms"] for r in runs)
        churn = statistics.median(r["churn"] for r in runs)
        rows.append((e2f, ttfs, churn))
        name = path.replace("\\", "/").split("/")[-1]
        print(f"{name[:40]:40s} {e2f:8.0f} ms {ttfs:6.0f} ms {churn:6.0f}")

    if not rows:
        return
    e2f = sorted(r[0] for r in rows)
    ttfs = sorted(r[1] for r in rows if r[1] == r[1])  # drop nan

    def pct(xs, p):
        if not xs:
            return float("nan")
        return xs[min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))]

    print("-" * 70)
    print(f"end→final : p50 {pct(e2f,50):.0f} ms   p95 {pct(e2f,95):.0f} ms   "
          f"(target ≤2000 ms; ≤1000 ms = full 25 pts)")
    print(f"ttfs      : p50 {pct(ttfs,50):.0f} ms   p95 {pct(ttfs,95):.0f} ms   "
          f"(target ≤1000 ms)")
    worst = pct(e2f, 95)
    verdict = ("baseline clears the target — speculative final likely unnecessary"
               if worst <= 2000 else
               "baseline misses ≤2000 ms at p95 — speculative final recommended")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
