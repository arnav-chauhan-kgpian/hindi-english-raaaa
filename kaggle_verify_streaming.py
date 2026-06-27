"""Kaggle (T4 GPU) verification for the Whisper-Hinglish streaming swap.

Run on a Kaggle GPU notebook:

    !git clone https://github.com/arnav-chauhan-kgpian/hindi-english-raaaa.git repo
    %cd repo
    !pip -q install -r requirements.txt
    !python kaggle_verify_streaming.py

Confirms the model LOADS, PRODUCES faithful Hinglish text, and the streaming draft()
end-to-final works — with per-clip latency. CUDA here is a proxy for the M1's MPS (the
transformers load path is device-independent), so this proves text quality + rough speed.
"""
import glob
import time
import wave

import numpy as np

from solution import transcribe as T
from solution import draft as D

SR = 16000
dev = "cuda" if T._cuda_available() else ("mps" if T._mps_available() else "cpu")
print(f"== device: {dev} | hinglish model: {T.HINGLISH_WHISPER_NAME} ==\n")


def read(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"{path}: need 16k"
        return w.readframes(w.getnframes())


# 1) load + direct Hinglish transcription (fidelity + per-call latency)
h = T.get_hinglish_model()
print(f"hinglish loaded: {h is not None} | err: {T._LAST_HINGLISH_ERROR or 'none'}")
print(f"meta: device={T._HINGLISH_META.get('device')} precision={T._HINGLISH_META.get('precision')} "
      f"load_ms={T._HINGLISH_META.get('load_ms')}\n")
print("== direct Whisper-Hinglish transcription ==")
for f in sorted(glob.glob("samples/*.wav")):
    audio = np.frombuffer(read(f), dtype="<i2").astype(np.float32) / 32768.0
    t = time.time()
    txt = T.hinglish_transcribe(audio).get("text", "")
    print(f"[{len(audio)/SR:4.1f}s {((time.time()-t)*1000):6.0f}ms] {f.split('/')[-1]}\n    {txt!r}")

# 2) streaming draft() end-to-final (the scored path)
print("\n== streaming draft() end-to-final ==")
D.warmup()
fb = int(0.02 * SR) * 2  # 20 ms frame in bytes
for f in sorted(glob.glob("samples/*.wav")):
    pcm = read(f)
    for end in range(fb, len(pcm) + 1, fb * 5):   # feed in ~100 ms steps
        D.draft(pcm[:end], False)
    t = time.time()
    final, sc = D.draft(pcm, True)
    print(f"[end->final {((time.time()-t)*1000):6.0f}ms, stable={sc}] {f.split('/')[-1]}\n    {final!r}")
print("\nDONE — if text is faithful Hinglish and latencies are low, the swap works.")
