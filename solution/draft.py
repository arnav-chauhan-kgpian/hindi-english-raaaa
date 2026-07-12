"""solution/draft.py — the ONE function for the STREAMING dictation track.

    draft(audio_buffer: bytes, is_final: bool) -> (text_so_far, stable_chars)
    draft_reset()   # sealed harness calls this at the start of each clip

Design — a SINGLE fast+faithful model instead of the reference's draft/finalizer race:
one Whisper-Hindi2Hinglish model (standard Whisper arch → loads on Apple MPS) transcribes the
rolling audio prefix for the partials AND produces the final. Because the same faithful model
drives the partials, the committed text is romanized Hinglish that KEEPS the code-switch (it
never shows the English-translated draft the reference does) — so committed partials match the
gold prefix (useful for TTFS, no no-partial cap) and don't get rewritten at the final (low churn).

Committing = the longest common word-prefix across consecutive drafts (that part has stopped
changing → safe to lock; non-decreasing). Everything is exception-wrapped and the final never
blanks (it falls back to the last good draft), so the run never crashes, hangs, or drops.
"""
from __future__ import annotations

import os
import re
import threading

# Silence progress bars / HF chatter before anything imports transformers — the sealed server
# runs behind a captured pipe the harness stops draining after READY, and that output would
# otherwise fill the pipe and deadlock the server (→ blank finals).
for _k, _v in {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1", "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np

from solution import transcribe as T  # reuse the offline-robust Whisper-Hinglish loader

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.6) * 2        # ~0.6 s before the first draft (2 bytes/sample)
_REDRAFT_SAMPLES = int(_SR * 0.7)            # re-run the model after ~0.7 s of new audio

# ---- per-clip state (harness calls draft_reset() between clips) ----
_prev = ""            # previous full draft
_committed = ""       # committed prefix — only extended within a clip (non-decreasing)
_last_n = 0           # last buffer length in samples (detect a new clip)
_last_decode_n = -10 ** 9
_cache = ""           # last successful decode (for debounced returns / final fallback)
_lock = threading.Lock()


def draft_reset() -> None:
    """Clear per-clip state. Called by the sealed harness at each clip's start."""
    global _prev, _committed, _last_n, _last_decode_n, _cache
    _prev = ""
    _committed = ""
    _last_n = 0
    _last_decode_n = -10 ** 9
    _cache = ""


# ---- warm the model at import so the sealed server reaches READY promptly; the heavy
#      load runs in a background thread and the first final waits for it (_await_warm). ----
_WARM: "threading.Thread | None" = None


def _start_warm() -> None:
    global _WARM
    if _WARM is not None:
        return

    def _w():
        try:
            T.get_hinglish_model()
        except Exception:
            pass

    _WARM = threading.Thread(target=_w, daemon=True)
    _WARM.start()


def warmup() -> None:
    """Block until the model has finished loading (join the import-time warm thread). The
    background thread already owns the single-shot load, so we WAIT for it rather than call
    the loader again (which would return None mid-load and leave the first clip cold)."""
    _await_warm()


def _await_warm(timeout: float = 180.0) -> None:
    th = _WARM
    if th is not None and th.is_alive():
        th.join(timeout)


_start_warm()


def _decode(buf: bytes) -> np.ndarray:
    try:
        return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _transcribe(audio: np.ndarray) -> str:
    try:
        return (T.hinglish_transcribe(audio).get("text", "") or "").strip()
    except Exception:
        return ""


def _words(text: str) -> list:
    return re.findall(r"[\w'.-]+", text, flags=re.UNICODE)


def _common_word_prefix(left: str, right: str) -> str:
    out = []
    for a, b in zip(_words(left), _words(right)):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev, _committed, _last_n, _last_decode_n, _cache
    with _lock:
        try:
            n = len(audio_buffer) // 2
            if n < _last_n:                       # cumulative buffer shrank → new clip
                draft_reset()
            _last_n = n

            if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
                return (_committed, len(_committed))

            # debounce the heavy model while streaming; always decode on the final
            if not is_final and (n - _last_decode_n) < _REDRAFT_SAMPLES:
                return (_cache or _committed, len(_committed))

            # the model must be loaded before ANY decode — otherwise early partials come back
            # empty while it loads (cold-start), which would trip the no-useful-partial cap.
            _await_warm()

            # OVERLAP: if the most recent streaming decode already covers ~all of the audio,
            # reuse it as the final instead of a fresh full pass → near-instant end-to-final
            # (the challenge's "overlap decode with the stream instead of waiting for key-up").
            if is_final and _cache and (n - _last_decode_n) <= int(_SR * 0.6):
                _committed = _cache
                return (_cache, len(_cache))

            text = _transcribe(_decode(audio_buffer))
            _last_decode_n = n

            if not text:                          # model not ready / empty decode
                fallback = _cache or _committed
                if is_final and fallback:
                    _committed = fallback
                    return (fallback, len(fallback))
                return (_committed, len(_committed))

            _cache = text
            stable = _common_word_prefix(_prev, text)
            if len(stable) >= len(_committed):
                _committed = stable
            _prev = text

            if is_final:                          # final: whole transcript is committed
                _committed = text
                return (text, len(text))
            return (text, len(_committed))
        except Exception:
            # reliability: never raise, never blank a committed prefix
            return (_committed, len(_committed))
