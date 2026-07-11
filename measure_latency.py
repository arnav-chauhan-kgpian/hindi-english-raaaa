"""Print REAL end-to-final latency per sample clip.

Run this on any Apple-silicon Mac with the accelerator on (the scoring box, the owner's Mac,
or a cloud EC2 mac instance). It reports the device actually used and the end-to-final time —
the number the GitHub CI cannot measure (its VM has no usable Metal).

    pip install -r requirements.txt -r requirements-streaming.txt
    python measure_latency.py

If `device: mps` and end-to-final is ~1-2s → good. If `device: cpu` → the box isn't giving us
the GPU (each decode ~15s) and we must fix the accelerator path, not the code.
"""
from __future__ import annotations

import glob
import os
import time
import wave

from solution import transcribe as T
from solution import draft as D

SR = 16000


def main() -> None:
    D.warmup()
    dev = T._HINGLISH_META.get("device")
    print(f"device: {dev}   model: {T.HINGLISH_WHISPER_NAME}\n")
    fb = int(0.02 * SR) * 2                       # 20 ms frame (bytes)
    e2f = []
    for wav in sorted(glob.glob("samples/*.wav")):
        with wave.open(wav, "rb") as w:
            pcm = w.readframes(w.getnframes())
        D.draft_reset()
        for e in range(fb * 25, len(pcm) + 1, fb * 25):   # ~500 ms feed steps (harness cadence)
            D.draft(pcm[:e], False)
        t = time.time()
        final, sc = D.draft(pcm, True)
        ms = (time.time() - t) * 1000
        e2f.append(ms)
        print(f"[end->final {ms:7.0f} ms] {os.path.basename(wav)}\n    {final[:80]}")
    if e2f:
        e2f.sort()
        print(f"\n  median end-to-final {e2f[len(e2f)//2]:.0f} ms  (target ~2000 ms)")
        print("  (device above must be 'mps' for this to be representative of the scoring box)")


if __name__ == "__main__":
    main()
