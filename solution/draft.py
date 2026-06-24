"""solution/draft.py — STREAMING dictation entry point (live track).

Contract (docs/STREAMING_CONTRACT.md):

    draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]

  * audio_buffer : cumulative PCM s16le, mono, 16 kHz, of everything heard so far
  * is_final     : False while streaming; True once the user stops
  * returns      : (text_so_far, stable_chars)
                   stable_chars = length of the committed prefix we promise not to rewrite

Design (Apple-silicon / M1, no CUDA — the frozen scoring box):
  * Streaming partials: faster-whisper-small (CPU int8) on the growing buffer, debounced.
    Commit a stable word-boundary prefix (monotonic) → low revision churn, fast TTFS.
  * Final (is_final): the SAME accuracy pipeline as the batch engine — recall router →
    Qwen3-ASR (Apple MPS) only for Hinglish → vocab/repair → Arabic strip. Routing is
    decided during streaming (sticky), so a Hinglish final goes straight to Qwen (skips a
    redundant fast pass) to keep end-to-final latency down.
  * Speculative final: when the speaker pauses near the end, the Qwen pass is started in the
    background (on MPS) BEFORE is_final arrives, so the final returns near-instantly. It is
    engineered to fail safe — it can never hang (timeout-bounded lock), never blank/crash
    (synchronous + committed-text fallbacks), never run two MPS calls at once (one lock), and
    never be slower than the plain synchronous path. Disable with STT_SPECULATIVE_FINAL=0.
  * Models warm in a background thread on the first call, so the load (one-time) does not
    block streaming. Everything is exception-wrapped — never blank-by-crash, never hang.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from solution import transcribe as T  # reuse models + the validated accuracy pipeline

TARGET_SR = 16000
_MIN_PARTIAL_S = 0.30          # need ≥300ms of audio before the first partial
_DEBOUNCE_S = 0.45            # re-run the fast model at most this often during streaming

# ---- speculative final: pre-run Qwen during the trailing pause so is_final lands fast ----
_SPEC_ENABLED = os.environ.get("STT_SPECULATIVE_FINAL", "1") not in ("0", "false", "False")
_SPEC_SILENCE_S = 0.35        # trailing window checked for a pause
_SPEC_SILENCE_RMS = 0.015     # tail RMS below this → likely end-of-utterance
_SPEC_MIN_GROWTH_S = 1.0      # don't relaunch unless ≥this much new audio since the last spec
_SPEC_COVER_S = 0.80          # spec is usable if the final buffer grew ≤this since the spec
_QWEN_FINAL_TIMEOUT = 12.0    # hard cap (s) on waiting for the Qwen lock at is_final → no hang

# ---- single active-stream state (the harness runs one clip at a time) -------
_LOCK = threading.Lock()
_STATE: dict = {}
_WARMED = False
_WARM_THREAD: Optional[threading.Thread] = None

_QWEN_LOCK = threading.Lock()  # serialize Qwen inference — never two MPS calls at once
_SPEC: tuple = (-1, "")        # (n_samples, raw_hinglish_text) — written via atomic rebind
_SPEC_RUNNING = False
_SPEC_GEN = 0                  # bumped on reset; a stale worker discards its result


def _reset() -> None:
    global _SPEC, _SPEC_GEN
    _SPEC = (-1, "")
    _SPEC_GEN += 1            # invalidate any in-flight speculative worker from a prior clip
    _STATE.clear()
    _STATE.update(committed_len=0, last_text="", last_len=0, last_run_len=-10 ** 9,
                  escalate=None, last_fast=None)


_reset()


def _warm_async() -> None:
    """Load fast + Hinglish models off the hot path (background thread)."""
    global _WARMED, _WARM_THREAD
    if _WARMED:
        return
    _WARMED = True

    def _w():
        try:
            T.get_fast_model()
        except Exception:
            pass
        try:
            # Warm Qwen too (MPS on the M1 scoring box, CPU elsewhere) so the load never
            # blocks the stream. get_hinglish_model self-gates (STT_DISABLE_CPU_QWEN).
            T.get_hinglish_model()
        except Exception:
            pass

    _WARM_THREAD = threading.Thread(target=_w, daemon=True)
    _WARM_THREAD.start()


def _await_warm(timeout: float = 120.0) -> None:
    """Block until the background warm thread finishes loading the models. Needed at
    is_final: the warm thread is the one caller that actually loads Qwen (the loader is
    single-shot), so the final must wait for it rather than race ahead and get None."""
    th = _WARM_THREAD
    if th is not None and th.is_alive():
        th.join(timeout)


def warmup() -> None:
    """Optional: pre-load the models before scoring (recommended on the frozen box)."""
    try:
        T.get_fast_model()
    except Exception:
        pass
    try:
        T.get_hinglish_model()   # self-gates on STT_DISABLE_CPU_QWEN
    except Exception:
        pass


def _decode(buf: bytes) -> np.ndarray:
    if not buf:
        return np.zeros(0, dtype=np.float32)
    try:
        return (np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _commit_len(prev: str, cur: str, prev_commit: int) -> int:
    """Stable prefix = longest common char prefix backed off to a word boundary; monotonic
    (never un-commits) → minimises revision churn while keeping a useful committed partial."""
    m = 0
    for a, b in zip(prev, cur):
        if a != b:
            break
        m += 1
    cut = cur.rfind(" ", 0, m)
    commit = cut + 1 if cut > 0 else 0
    return max(0, min(len(cur), max(commit, prev_commit)))


def _fast_text(audio: np.ndarray):
    """Run the fast recognizer; returns the result dict or None."""
    try:
        if T.get_fast_model() is None:
            return None
        return T.fast_transcribe(audio)
    except Exception:
        return None


def _trailing_silence(audio: np.ndarray) -> bool:
    """True if the last ~350 ms look like a pause (likely end-of-utterance)."""
    k = int(_SPEC_SILENCE_S * TARGET_SR)
    if audio.shape[0] < k:
        return False
    try:
        tail = audio[-k:]
        return float(np.sqrt(np.mean(tail * tail))) < _SPEC_SILENCE_RMS
    except Exception:
        return False


def _qwen_raw(audio: np.ndarray, timeout: Optional[float] = None) -> Optional[str]:
    """Run Qwen once, serialized via _QWEN_LOCK (one MPS call at a time). Returns the raw
    transcript, ``""`` if the model is unavailable/blank, or ``None`` if the lock could not
    be acquired within ``timeout`` — the caller then degrades instead of blocking/hanging."""
    acquired = _QWEN_LOCK.acquire(timeout=timeout) if timeout is not None else _QWEN_LOCK.acquire()
    if not acquired:
        return None
    try:
        if T.get_hinglish_model() is None:
            return ""
        return T.hinglish_transcribe(audio).get("text", "")
    except Exception:
        return ""
    finally:
        _QWEN_LOCK.release()


def _spec_worker(audio: np.ndarray, n: int, gen: int) -> None:
    """Background: compute the Qwen final on a snapshot so the eventual is_final is instant."""
    global _SPEC, _SPEC_RUNNING
    try:
        txt = _qwen_raw(audio)             # background → fine to wait for the lock
        if txt and gen == _SPEC_GEN:       # discard if a new clip started meanwhile
            _SPEC = (n, txt)
    except Exception:
        pass
    finally:
        _SPEC_RUNNING = False


def _maybe_speculate(audio: np.ndarray, n: int) -> None:
    """If the speaker paused near the end (Hinglish clip), pre-run the Qwen final in the
    background. Best-effort and fully optional — if it never fires, is_final just runs the
    plain synchronous path. Never blocks the caller."""
    global _SPEC_RUNNING
    if not _SPEC_ENABLED or _SPEC_RUNNING:
        return
    if _STATE.get("escalate") is not True:
        return
    if (n - _SPEC[0]) < int(_SPEC_MIN_GROWTH_S * TARGET_SR):
        return
    if not _trailing_silence(audio):
        return
    try:
        if T.get_hinglish_model() is None:   # model not ready/available → don't speculate
            return
    except Exception:
        return
    _SPEC_RUNNING = True
    threading.Thread(target=_spec_worker, args=(audio, n, _SPEC_GEN), daemon=True).start()


def _finalize(audio: np.ndarray) -> str:
    """Best faithful final on the complete buffer — the batch accuracy pipeline."""
    n = int(audio.shape[0])         # samples in the complete buffer (used by the spec-cover check)
    escalate = _STATE.get("escalate")
    fast_text = ""
    try:
        # Hinglish was already decided during streaming → skip a redundant fast pass and go
        # straight to Qwen (lower end-to-final latency). Otherwise run fast once on the full buffer.
        if escalate is None:
            fr = _fast_text(audio) or {}
            fast_text = fr.get("text", "")
            escalate = bool(fr) and T.should_escalate(
                fr.get("language", ""), float(fr.get("language_probability", 0.0) or 0.0),
                float(fr.get("avg_logprob", 0.0) or 0.0),
                float(fr.get("compression_ratio", 0.0) or 0.0), fast_text)
        elif not escalate:
            fr = _fast_text(audio) or {}
            fast_text = fr.get("text", "")

        hing_text = ""
        if escalate:
            _await_warm()   # ensure the (single-shot) Qwen load finished before we use it
            sl, st = _SPEC  # snapshot the speculative result (atomic tuple read)
            if _SPEC_ENABLED and st and 0 <= sl <= n and (n - sl) <= int(_SPEC_COVER_S * TARGET_SR):
                hing_text = st            # pause-time speculation already covers this buffer → instant
            else:
                # no usable speculation → run Qwen now (timeout-bounded lock → never hangs;
                # None means the lock was stuck, so we degrade to the fast/committed text).
                txt = _qwen_raw(audio, timeout=_QWEN_FINAL_TIMEOUT)
                hing_text = txt or ""
            # Qwen unavailable / blank → we skipped the fast pass; run it now on the full
            # buffer so the final still captures everything (incl. the tail), never blank.
            if not hing_text.strip() and not fast_text.strip():
                fast_text = (_fast_text(audio) or {}).get("text", "")

        final = T.finalize_transcript(hing_text, fast_text, "auto", bool(hing_text.strip()))
        if T._vocab is not None:
            if T.ENABLE_VOCAB:
                final = T._apply_preserving_hindi(final, T._vocab.normalize_tech_words)
            if T.ENABLE_REPAIR:
                final = T._apply_preserving_hindi(final, T._vocab.repair_common_asr_errors)
        final = T._strip_arabic_if_hinglish(final, bool(hing_text.strip()))
        final = (final or "").strip()
        # never return blank if we have anything committed/recognized
        return final or _STATE.get("last_text", "").strip() or fast_text.strip()
    except Exception:
        return _STATE.get("last_text", "").strip()


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """Streaming dictation entry point. See docs/STREAMING_CONTRACT.md."""
    _warm_async()
    with _LOCK:
        try:
            audio = _decode(audio_buffer)
            n = int(audio.shape[0])

            # new clip → the cumulative buffer shrank
            if n < _STATE.get("last_len", 0):
                _reset()
            _STATE["last_len"] = n

            # ---- FINAL ----
            if is_final:
                final = _finalize(audio)
                _reset()
                return (final, len(final))

            # ---- STREAMING PARTIAL ----
            if n < int(_MIN_PARTIAL_S * TARGET_SR):
                txt = _STATE.get("last_text", "")
                return (txt, _STATE.get("committed_len", 0))

            # debounce: re-run only after enough new audio
            if (n - _STATE.get("last_run_len", -10 ** 9)) < int(_DEBOUNCE_S * TARGET_SR):
                return (_STATE.get("last_text", ""), _STATE.get("committed_len", 0))

            fr = _fast_text(audio)
            _STATE["last_run_len"] = n
            if not fr or not fr.get("text", "").strip():
                return (_STATE.get("last_text", ""), _STATE.get("committed_len", 0))

            text = fr["text"].strip()
            _STATE["last_fast"] = fr
            # sticky routing decision (so the final can skip the fast pass for Hinglish)
            if _STATE.get("escalate") is not True:
                try:
                    esc = T.should_escalate(
                        fr.get("language", ""), float(fr.get("language_probability", 0.0) or 0.0),
                        float(fr.get("avg_logprob", 0.0) or 0.0),
                        float(fr.get("compression_ratio", 0.0) or 0.0), text)
                    _STATE["escalate"] = True if esc else (_STATE.get("escalate") or False)
                    # (Qwen is already warming in the background thread from the first call;
                    # don't load it here — on CPU that would block this partial.)
                except Exception:
                    pass

            commit = _commit_len(_STATE.get("last_text", ""), text, _STATE.get("committed_len", 0))
            _STATE["committed_len"] = commit
            _STATE["last_text"] = text
            _maybe_speculate(audio, n)   # pre-run the Qwen final if the speaker paused
            return (text, commit)
        except Exception:
            # reliability: never raise out of the streaming hot path
            return (_STATE.get("last_text", ""), _STATE.get("committed_len", 0))


# Warm the models at import time (best-effort, background). The scored run is offline
# (network blocked after warmup) — loading here, when the harness imports the module during
# its network-up setup phase, populates the cache so the first scored clip is already hot.
try:
    _warm_async()
except Exception:
    pass


if __name__ == "__main__":
    # smoke test with a sample wav (decoded to PCM s16le) — no harness needed.
    import sys, wave, os
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "samples", "openslr104_hi_en_103085_w5Jyq3XMbb3WwiKQ_0000.wav")
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    assert sr == TARGET_SR, f"expected 16kHz, got {sr}"
    # feed in 200ms steps to mimic streaming
    step = int(0.2 * TARGET_SR) * 2  # bytes (s16le)
    for end in range(step, len(pcm) + 1, step):
        t, sc = draft(pcm[:end], False)
        print(f"partial[{end//2/TARGET_SR:5.1f}s] stable={sc:3d} | {t[:80]}")
    t0 = time.time()
    final, sc = draft(pcm, True)
    print(f"\nFINAL ({(time.time()-t0)*1000:.0f} ms, stable={sc}):\n{final}")
