"""Offline STREAMING preview — mirrors the streaming scoring path.

Feeds each clip in samples/ (or data/dev/audio) through solution.draft.draft() in real-time
20 ms frames with the network hard-blocked, and prints the committed final + end-to-final
latency per clip. This is the streaming counterpart to preview.py (which runs the batch
transcribe()). Run AFTER the model cache is warmed once with network available.

    python preview_stream.py
"""
from __future__ import annotations

import glob
import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from offline_guard import block_network
from solution import draft as D

SR = 16000


def _read(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise ValueError(f"{path}: need 16 kHz mono, got {w.getframerate()}Hz {w.getnchannels()}ch")
        return w.readframes(w.getnframes())


def main() -> None:
    clips = sorted(glob.glob(os.path.join(HERE, "samples", "*.wav")))
    if not clips:
        clips = sorted(glob.glob(os.path.join(HERE, "data/dev/audio", "*.wav")))
    if not clips:
        print("no sample clips found"); return

    D.warmup()          # load models from the warmed cache BEFORE the network is blocked
    block_network()     # mirror official scoring: no network during the timed run

    frame = int(0.02 * SR) * 2   # 20 ms frame in bytes (PCM s16le)
    lat, blanks = [], 0
    for path in clips:
        try:
            pcm = _read(path)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}"); continue
        for end in range(frame, len(pcm) + 1, frame * 5):   # ~100 ms feed steps
            D.draft(pcm[:end], False)
        t0 = time.time()
        final, stable = D.draft(pcm, True)
        ms = (time.time() - t0) * 1000
        lat.append(ms)
        if not (final or "").strip():
            blanks += 1
        print(f"[end->final {ms:7.0f} ms  stable={stable:4d}] {os.path.basename(path)}\n    {final!r}")

    if lat:
        lat.sort()
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))]
        print(f"\n  clips {len(lat)}  end->final p50 {p50:.0f} ms  p95 {p95:.0f} ms  blanks {blanks}/{len(lat)}")
        print("  (target: clean final under ~2000 ms; blanks must be 0)")


if __name__ == "__main__":
    main()
