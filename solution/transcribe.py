"""builderr · local dual-language dictation engine — submission entry point.

Architecture (all inside this single file, fully local, no cloud at scoring):

    audio ─► preprocess ─► fast ASR (faster-whisper-small-int8, GPU)
          ─► recall router ─► [escalate?] ─► Hinglish ASR (Qwen3-ASR 0.6B, GPU vLLM/Triton)
          ─► finalizer (faithful, no translation/romanization; repetition + blank guards)
          ─► JSON contract

Contract (REQUIRED — checked by the harness):

    python -m solution.transcribe --input clip.wav --mode auto --output result.json

and the importable function ``transcribe(wav_path, mode) -> dict`` used by preview.py.

Design rules enforced here:
  * Linux-CPU + macOS compatible (no MLX / CoreML / Apple-only paths — the scoring box is Linux).
  * Offline after warmup: models load with ``local_files_only=True`` first; a network download is
    only attempted as a warmup fallback when the cache is cold AND the net is still open.
  * Commercial-friendly models only (faster-whisper / Whisper weights, Apache-2.0 Qwen3-ASR).
  * Code-switch is kept faithful — ``task="transcribe"`` everywhere, NEVER ``translate``.
  * Every entry point is wrapped so no exception escapes; the contract always validates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Optional, Union

# --- model-loading policy -----------------------------------------------------
# We DO NOT force HF_HUB_OFFLINE=1 here. Forcing it previously blocked the very first
# download, so the cache stayed empty and every model load returned None forever.
# Instead the loaders try the local cache first and only attempt a download when the
# network is actually reachable AND not blocked (see _downloads_allowed). After a clip
# is cached, subsequent loads hit the local cache (offline). Set STT_OFFLINE=1 (or
# HF_HUB_OFFLINE=1) to force pure-offline; set STT_DEBUG=1 for verbose load logging.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Apple-silicon robustness: if any model op isn't implemented on the Metal (MPS) backend,
# run THAT op on CPU instead of raising. Turns a hard MPS error into a graceful per-op
# fallback — important on the M1 box where the Qwen path runs on MPS. (No effect off Apple.)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

VERBOSE = os.environ.get("STT_DEBUG") == "1"


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[stt] {msg}", file=sys.stderr)

# Audio array type is numpy; we keep it loosely typed so the module imports even if
# numpy is briefly unavailable (transcribe() then falls back to passing the file path).
AudioInput = Union["Any", str]

# --- model identifiers (surfaced in model_ids / raw_candidates for auditability)
FAST_MODEL_ID = "faster-whisper-small-int8"
HINGLISH_QWEN_ID = "qwen3-asr-0.6b-hinglish"           # legacy (custom arch — did not load on the M1)
HINGLISH_WHISPER_ID = "whisper-hindi2hinglish"         # active: standard Whisper arch, loads on MPS

FAST_MODEL_NAME = "small"
HINGLISH_QWEN_NAME = "moorlee/qwen3-asr-0.6b-hinglish"
# Hinglish final = a Whisper finetune (standard transformers arch → loads cleanly on Apple
# MPS, unlike the custom qwen3_asr arch). Default = Apex (~800M): on a Kaggle T4 it matched
# Prime's (large-v3) Hinglish fidelity while running ~4x faster (end-to-final 0.4–1.2s vs
# 1.4–5.1s), which keeps it under the latency caps on the slower M1 GPU. Override via env to
# trade latency vs fidelity: Prime (2B, max fidelity) | Apex (~800M, balanced) | Swift (72M).
HINGLISH_WHISPER_NAME = os.environ.get("STT_HINGLISH_MODEL", "Oriserve/Whisper-Hindi2Hinglish-Apex")

# Generation cap for Qwen3-ASR. Decode is ~95% of latency and ~linear in tokens; typical
# dictation clips stop at EOS well under this, so 256 is quality-safe and only bounds
# pathological runaway (halves the 512 worst case). Greedy/num_beams=1/do_sample=False are
# the qwen-asr CPU defaults (latency-optimal). Override via STT_QWEN_MAX_NEW_TOKENS.
QWEN_MAX_NEW_TOKENS = int(os.environ.get("STT_QWEN_MAX_NEW_TOKENS", "256"))

TARGET_SR = 16000

# --- vocab module (tech-term normalization + ASR repair); graceful if absent ----
try:
    from solution import vocab as _vocab
except Exception:  # pragma: no cover - package-relative fallback
    try:
        from . import vocab as _vocab
    except Exception:
        _vocab = None

# --- pipeline stage flags (FROZEN for submission) ----
ENABLE_ROUTER = True       # route English vs escalate (should_escalate)
ENABLE_HINGLISH = True     # run the Hinglish ASR on escalated clips
ENABLE_ENSEMBLE = False    # PERMANENTLY DISABLED — ensemble merge removed (do not enable)
ENABLE_VOCAB = True        # apply vocab.normalize_tech_words to the final text
ENABLE_REPAIR = True       # apply vocab.repair_common_asr_errors to the final text

# =============================================================================
# PART 1 — module-level singletons + fast-model loader
# =============================================================================
FAST_MODEL: Optional[Any] = None
HINGLISH_MODEL: Optional["_HinglishHandle"] = None

_FAST_WARMED = False
_FAST_TRIED = False        # cache load failure so we don't re-attempt (and re-hang) per clip
_HINGLISH_TRIED = False


def _cpu_threads() -> int:
    try:
        return max(1, (os.cpu_count() or 4))
    except Exception:
        return 4


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    """Apple-silicon Metal (M1/M2/M3) accelerator."""
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _gpu_available() -> bool:
    """True if any accelerator (CUDA or Apple MPS) is usable. Selects the fastest Qwen
    backend; the model still runs on CPU as a fidelity-first fallback (see get_hinglish_model)."""
    return _cuda_available() or _mps_available()


def _cpu_qwen_allowed() -> bool:
    """Run the Hinglish model on CPU when no accelerator exists? Default YES (fidelity-first:
    code-switch quality > latency on a CPU box). Opt out with STT_DISABLE_CPU_QWEN=1 to keep
    the final fast (faster-whisper only) on machines without a GPU."""
    return os.environ.get("STT_DISABLE_CPU_QWEN", "0") not in ("1", "true", "True")


def _gpu_dtype():
    """bf16 on Ampere+ CUDA; fp16 on T4/Turing and on Apple MPS."""
    import torch
    try:
        if _cuda_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


# --- load-policy diagnostics (populated by the loaders; read by model_debug_status) ---
_LAST_FAST_ERROR = ""
_LAST_HINGLISH_ERROR = ""
_FAST_META: dict = {}
_HINGLISH_META: dict = {}


def _hf_cache_dir() -> str:
    return (os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def _network_is_blocked() -> bool:
    """True if offline_guard.block_network() has patched the socket (scored run)."""
    try:
        import socket
        return getattr(socket.socket.connect, "__name__", "") == "guarded_connect"
    except Exception:
        return False


def _can_reach_hf(timeout: float = 3.0) -> bool:
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except Exception:
        return False


def _call_with_timeout(fn, timeout: Optional[float]):
    """Run fn() with a wall-clock cap. Raises TimeoutError if exceeded. ``timeout=None``
    runs without a cap. Used so an offline/blocked load can never hang (a partial cache
    otherwise retries the blocked network for tens of seconds)."""
    if timeout is None:
        return fn()
    import threading
    box: dict = {}

    def _run() -> None:
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001 — propagate to caller
            box["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"model load exceeded {timeout:.0f}s (likely a partial/blocked cache)")
    if "e" in box:
        raise box["e"]
    return box.get("r")


def _load_timeout() -> Optional[float]:
    """No cap when a download is expected (legit and slow); a hard cap otherwise so an
    offline/blocked load with a partial cache fails fast instead of hanging."""
    return None if _downloads_allowed() else 15.0


def _downloads_allowed() -> bool:
    """A one-time download is OK only when not explicitly offline, not behind the
    network guard, and the hub is actually reachable — otherwise we'd hang."""
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        return False
    if os.environ.get("STT_OFFLINE") == "1":
        return False
    if _network_is_blocked():
        return False
    return _can_reach_hf()


def _load_faster_whisper(model_name: str, meta: Optional[dict] = None) -> Any:
    """Load a faster-whisper (CTranslate2) model on CPU. Tries the LOCAL cache first
    (offline, instant) and only attempts a one-time download when ``_downloads_allowed``.
    Raises if every attempt fails (caller records the error)."""
    from faster_whisper import WhisperModel  # local import: keeps module import cheap

    threads = _cpu_threads()
    last_err: Optional[Exception] = None
    allow_dl = _downloads_allowed()
    # local cache first; only add the download attempt when it's genuinely allowed
    local_only_opts = (True, False) if allow_dl else (True,)
    # CTranslate2 supports CUDA or CPU only (no Apple MPS) — so try CUDA when present, else
    # CPU int8 (fast on Apple-silicon ARM CPUs). CPU fallback is always available.
    attempts = []
    if _cuda_available():
        attempts += [("cuda", "int8_float16"), ("cuda", "float16")]
    attempts += [("cpu", "int8"), ("cpu", "int8_float32"), ("cpu", "float32")]
    _log(f"faster-whisper '{model_name}': attempts={attempts} downloads_allowed={allow_dl} "
         f"local_only_opts={local_only_opts} cache={_hf_cache_dir()}")
    for local_only in local_only_opts:
        for device, compute_type in attempts:
            try:
                m = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=threads,
                    num_workers=1,
                    local_files_only=local_only,
                )
                if meta is not None:
                    meta.update(model=model_name, compute_type=compute_type, device=device,
                                local_files_only=local_only, cache=_hf_cache_dir())
                _log(f"faster-whisper '{model_name}' LOADED "
                     f"(compute_type={compute_type}, local_files_only={local_only})")
                return m
            except Exception as e:  # noqa: BLE001 — try the next combination
                last_err = e
    raise RuntimeError(f"could not load faster-whisper '{model_name}': "
                       f"{type(last_err).__name__}: {last_err}")


def get_fast_model() -> Optional[Any]:
    """Return the resident fast multilingual recognizer (faster-whisper small, int8).

    Loaded once, warmed once (a tiny silent decode so the first real clip is hot),
    and cached at module scope. Returns ``None`` if the engine cannot be loaded so
    the caller can degrade gracefully instead of crashing.
    """
    global FAST_MODEL, _FAST_WARMED, _FAST_TRIED, _LAST_FAST_ERROR
    if FAST_MODEL is not None:
        return FAST_MODEL
    if _FAST_TRIED:            # already failed once — don't retry the (possibly slow) load
        return None
    _FAST_TRIED = True
    try:
        FAST_MODEL = _call_with_timeout(
            lambda: _load_faster_whisper(FAST_MODEL_NAME, _FAST_META), _load_timeout())
        _LAST_FAST_ERROR = ""
    except Exception as e:
        _LAST_FAST_ERROR = f"{type(e).__name__}: {e}"
        _log(f"FAST MODEL not loaded — {_LAST_FAST_ERROR}")
        FAST_MODEL = None
        return None

    if not _FAST_WARMED:
        try:
            import numpy as np

            silence = np.zeros(TARGET_SR // 2, dtype=np.float32)  # 0.5s of silence
            seg_iter, _ = FAST_MODEL.transcribe(silence, beam_size=1, vad_filter=False)
            for _ in seg_iter:  # drain the generator to force the decode
                pass
        except Exception:
            pass  # warmup is best-effort; a cold first decode is still correct
        _FAST_WARMED = True
    return FAST_MODEL


# =============================================================================
# PART 2 — audio loading / preprocessing
# =============================================================================
def _read_audio(path: str):
    """Decode wav/flac/mp3 to (float32 samples, sample_rate). Tries soundfile then
    librosa (covers mp3 via audioread). Returns (np.ndarray, int)."""
    import numpy as np

    last_err: Optional[Exception] = None
    # 1) libsndfile (wav/flac, and mp3 on recent builds)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        last_err = e
    # 2) librosa (broad format support incl. mp3); keep native sr, mono handled below
    try:
        import librosa

        data, sr = librosa.load(path, sr=None, mono=False)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        last_err = e
    raise RuntimeError(f"could not decode audio '{path}': {last_err}")


def _to_mono(y):
    import numpy as np

    if y.ndim > 1:
        # soundfile → (frames, channels); librosa → (channels, frames). Reduce the channel axis.
        axis = 1 if y.shape[0] >= y.shape[-1] else 0
        y = y.mean(axis=axis)
    return np.asarray(y, dtype=np.float32).reshape(-1)


def _resample(y, sr: int, target: int = TARGET_SR):
    if sr == target or y.size == 0:
        return y
    import numpy as np

    try:  # high-quality polyphase resample when scipy is present
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target))
        return resample_poly(y, target // g, sr // g).astype(np.float32)
    except Exception:
        # linear-interpolation fallback (no scipy dependency)
        n_target = int(round(y.shape[0] * float(target) / float(sr)))
        if n_target <= 1:
            return y.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=y.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32)


def _normalize(y):
    import numpy as np

    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = (y / peak).astype(np.float32)
    return y


def _trim_silence(y, sr: int = TARGET_SR):
    """Best-effort energy-based VAD trim. Never trims to empty; returns the input on
    any failure or when no voiced frame is found."""
    try:
        import numpy as np

        if y.size < sr // 5:  # < 0.2s — nothing to trim
            return y
        frame = max(1, int(0.02 * sr))  # 20ms frames
        n = (y.shape[0] // frame) * frame
        if n < frame:
            return y
        frames = y[:n].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        thr = max(1e-3, 0.05 * float(np.max(rms)))
        voiced = np.where(rms >= thr)[0]
        if voiced.size == 0:
            return y
        pad = 5  # keep ~100ms of context on each side
        start = max(0, (int(voiced[0]) - pad)) * frame
        end = min(n, (int(voiced[-1]) + 1 + pad) * frame)
        trimmed = y[start:end]
        return trimmed if trimmed.size > 0 else y
    except Exception:
        return y


def load_audio(path: str, do_vad: bool = True):
    """Load ``path`` → mono, 16kHz, peak-normalized float32 numpy array.

    Supports wav/flac/mp3. Optional energy-based VAD trimming. Returns ``None`` on
    failure so the caller can fall back to letting the ASR engine decode the file
    path directly.
    """
    try:
        y, sr = _read_audio(path)
        y = _to_mono(y)
        y = _resample(y, sr, TARGET_SR)
        y = _normalize(y)
        if do_vad:
            y = _trim_silence(y, TARGET_SR)
        import numpy as np

        return np.ascontiguousarray(y, dtype=np.float32)
    except Exception:
        return None


# =============================================================================
# PART 3 — fast transcription + segment-stat aggregation
# =============================================================================
def _aggregate_segments(seg_iter) -> dict[str, Any]:
    """Drain a faster-whisper segment generator into text + duration-weighted stats."""
    texts: list[str] = []
    dur_total = 0.0
    alp_acc = cr_acc = nsp_acc = 0.0
    for seg in seg_iter:
        txt = getattr(seg, "text", "") or ""
        texts.append(txt)
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        d = max(end - start, 1e-3)
        dur_total += d
        alp_acc += float(getattr(seg, "avg_logprob", 0.0) or 0.0) * d
        cr_acc += float(getattr(seg, "compression_ratio", 0.0) or 0.0) * d
        nsp_acc += float(getattr(seg, "no_speech_prob", 0.0) or 0.0) * d
    text = " ".join(t.strip() for t in texts).strip()
    text = re.sub(r"\s+", " ", text)
    if dur_total <= 0.0:
        return {"text": text, "avg_logprob": 0.0, "compression_ratio": 0.0,
                "no_speech_prob": 1.0 if not text else 0.0}
    return {
        "text": text,
        "avg_logprob": alp_acc / dur_total,
        "compression_ratio": cr_acc / dur_total,
        "no_speech_prob": nsp_acc / dur_total,
    }


def fast_transcribe(audio: AudioInput) -> dict[str, Any]:
    """Run the fast multilingual recognizer. ``audio`` is a 16kHz float32 array or a
    file path. Returns text + the router's sensor signals + elapsed ms."""
    t0 = time.time()
    out: dict[str, Any] = {
        "text": "", "language": "", "language_probability": 0.0,
        "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 1.0, "time_ms": 0,
    }
    model = get_fast_model()
    if model is None:
        out["time_ms"] = round((time.time() - t0) * 1000)
        return out
    try:
        seg_iter, info = model.transcribe(
            audio,
            language=None,                 # detect → drives the router
            task="transcribe",             # NEVER translate
            beam_size=1,                   # greedy: latency lever
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        agg = _aggregate_segments(seg_iter)
        out.update(agg)
        out["language"] = str(getattr(info, "language", "") or "")
        out["language_probability"] = float(getattr(info, "language_probability", 0.0) or 0.0)
    except Exception:
        pass  # leave the safe defaults; the router will escalate on the empty text
    out["time_ms"] = round((time.time() - t0) * 1000)
    return out


# =============================================================================
# PART 5 — Devanagari detection (used by the router + language guess)
# =============================================================================
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def contains_devanagari(text: str) -> bool:
    """True if ``text`` contains any Devanagari codepoint (U+0900–U+097F)."""
    return bool(_DEVANAGARI.search(text or ""))


# =============================================================================
# PART 4 — router  (recall-biased)
#
# Theory: a false negative (a code-switch clip routed to the fast English path) is
# catastrophic — the fast path can translate the mix away, which both loses meaning
# and trips the scorecard's WER>0.9 / critical-flip caps (20–50). A false positive
# (escalating clean English) only costs ~1s of latency. So we escalate on the UNION
# of many cheap signals and lower every threshold toward recall.
#
# The hardest miss the old router made was *romanized* Hinglish ("rollback abhi mat
# karo pehle p95 check karlo"): Whisper writes it in Latin, tags it English with high
# probability and good logprob, and there is no Devanagari — so every numeric signal
# says "clean English". A lexical detector (detect_hinglish_score) is the only cheap
# way to catch it, so it is the centerpiece of the new router.
# =============================================================================

# Indic language tags faster-whisper may emit (Hindi is frequently tagged as Urdu).
_INDIC_LANGS = frozenset({"hi", "ur", "ne", "mr", "bn", "pa", "gu", "sa", "kok", "or", "as"})

# Work/tech terms that, when mixed with Hindi, signal builder code-switch speech.
_TECH_HINTS = frozenset({
    "aws", "gpt", "docker", "kubernetes", "k8s", "jira", "cursor", "prd", "api", "kafka",
    "redis", "codex", "github", "git", "pr", "ci", "cd", "deploy", "rollback", "latency",
    "p95", "p99", "server", "db", "database", "cache", "prod", "staging", "bug", "merge",
    "commit", "endpoint", "model", "prompt", "token", "gpu", "cpu", "python", "node",
    "react", "sql", "json", "backend", "frontend", "build", "release", "ticket", "sprint",
})

# Romanized Hindi / Hinglish lexicon (~250 distinctive tokens). English-colliding words
# (to / do / the / is / me / main / so / hi / are / on / in / it / by ...) are deliberately
# EXCLUDED so the detector still discriminates; everything here is rare in English text.
HINGLISH_KEYWORDS = frozenset({
    # seed list
    "hai", "nahi", "abhi", "karo", "karlo", "mat", "mera", "tum", "kyu", "kyunki", "acha",
    "bhai", "pehle", "wala", "wali", "sab", "kuch", "yaar", "haan", "nhi",
    # pronouns / possessives
    "mai", "mein", "mujhe", "meri", "mere", "tu", "tera", "teri", "tere", "tujhe", "tumhe",
    "tumhare", "tumhari", "tumne", "aap", "aapka", "aapko", "aapki", "hum", "humein", "hamein",
    "humara", "hamara", "hamare", "hamari", "woh", "wo", "voh", "yeh", "ye", "inka", "unka",
    "inko", "unko", "iska", "uska", "iski", "uski", "inke", "unke", "kisi", "kisne", "kisko",
    "koi", "kuchh", "sabko", "sabhi", "apna", "apni", "apne", "khud", "usne", "maine", "hamne",
    "tumlog", "humlog",
    # be / aux / very common verbs and inflections
    "hain", "ho", "hu", "hoon", "hun", "tha", "thi", "raha", "rahi", "rahe", "hoga",
    "hogi", "honge", "hota", "hoti", "hote", "hone", "karu", "karun", "karna", "karne", "karta",
    "karti", "karte", "kiya", "kiye", "kare", "karenge", "karke", "karwao", "kar", "lo", "le",
    "lena", "liya", "lega", "legi", "denge", "dena", "diya", "de", "dijiye", "jao", "jana",
    "gaya", "gayi", "gaye", "aao", "aana", "aaya", "aaye", "dekho", "dekhna", "dekha", "suno",
    "sun", "bolo", "bola", "kaho", "kehna", "kaha", "milega", "mila", "milta", "chahiye",
    "chaiye", "chahta", "chahti", "sakta", "sakti", "sakte", "banao", "banana", "banaya",
    "samajh", "samjha", "samjho", "samajhna", "padega", "padta", "lagta", "lagi", "laga",
    "lagao", "rakho", "rakh", "rakhna", "bhejo", "bhej", "bhejna", "batao", "bata", "batana",
    "pucho", "puchna", "poochna", "likho", "likh", "likhna", "padho", "padhna", "sikho",
    "seekhna", "samjhao", "hatao", "hata", "daalo", "daal", "nikalo", "nikal", "utha", "uthao",
    "baith", "baitho", "ruko", "ruk", "rukja", "chodo", "chhodo", "chod", "pakdo", "pakad",
    "kholo", "khol", "band", "chalao", "chala", "milao", "laao",
    # time / adverbs
    "kal", "aaj", "aj", "parso", "pehla", "pehli", "baad", "fir", "phir", "ab", "tab", "jab",
    "kab", "yahan", "yaha", "wahan", "waha", "kahan", "kaha", "idhar", "udhar", "kidhar", "andar",
    "bahar", "upar", "uper", "niche", "neeche", "saath", "sath", "bina", "tak", "sirf", "bilkul",
    "zyada", "jyada", "thoda", "thodi", "bahut", "bohot", "bhot", "kafi", "kaafi", "jaldi",
    "dhire", "dheere", "turant", "hamesha", "kabhi", "roz", "dobara", "wapas", "waapas",
    "filhaal", "filhal", "abhitak", "tabtak", "jabtak",
    # negation / question / connectors
    "nahin", "na", "kyun", "kyon", "kyonki", "kaise", "kaisa", "kaisi", "kya", "kyaa", "kitna",
    "kitni", "kitne", "kaun", "kon", "kaunsa", "konsa", "matlab", "agar", "warna", "lekin",
    "magar", "aur", "toh", "phirbhi", "par", "pe", "ki", "ka", "ke", "ko", "se", "bhi", "jo",
    "jise", "jiska", "wajah", "kyunke",
    # adjectives / nouns / fillers
    "accha", "achha", "achhi", "achhe", "theek", "thik", "sahi", "galat", "bura", "buri", "naya",
    "nayi", "purana", "purani", "bada", "badi", "chota", "choti", "chhota", "chhoti", "kaam",
    "baat", "baatein", "cheez", "cheezein", "log", "logo", "logon", "paisa", "paise", "ghar",
    "din", "raat", "samay", "waqt", "saal", "mahina", "mahine", "dost", "behen", "beta", "didi",
    "bhaiya", "bhaiyya", "sahab", "saheb", "ji", "han", "ha", "hmm", "arre", "arrey", "oye",
    "chalo", "chal", "bas", "jaisa", "jaise", "jaisi", "waisa", "waise", "itna", "itni", "utna",
    "aisa", "aise", "aisi", "vaise", "shayad", "sayad", "zaroor", "zarur", "jarur", "shukriya",
    "dhanyavad", "namaste", "namaskar", "swagat", "kripya", "maaf", "maafi", "wale", "waala",
    "waale", "khana", "peena", "yaani", "yani", "waqai", "sacchi", "sach", "jhooth", "gussa",
    "khush", "pareshan", "dhyan", "mast", "ekdum", "vagairah",
})

HINGLISH_SCORE_THRESHOLD = 0.15  # low on purpose — recall over precision

_LATIN_WORD = re.compile(r"[A-Za-z]+")


def detect_hinglish_score(
    text: str,
    language: str = "",
    language_probability: float = 1.0,
    avg_logprob: float = 0.0,
) -> float:
    """Lexical+statistical Hinglish likelihood in [0, 1].

    Features (each normalized to [0,1], then a weighted sum, clamped):
      f_deva  : Devanagari character ratio          (script evidence)
      f_kw    : Hindi-keyword ratio over Latin words (romanized code-switch)
      f_kwc   : Hindi-keyword count (saturating at 3)
      f_mixed : both scripts present                 (definite code-switch)
      f_tech  : tech term present AND any Hindi       (builder code-switch speech)
      f_len   : short average word length            (romanized Hindi skews short)
      f_lang  : non-English / low language probability
      f_conf  : low avg_logprob                      (decoder uncertainty)
    """
    t = text or ""
    if not t.strip():
        return 0.0

    latin = [w.lower() for w in _LATIN_WORD.findall(t)]
    n_latin = len(latin)
    n_latin_chars = sum(len(w) for w in latin)
    n_deva = len(_DEVANAGARI.findall(t))
    total_alpha = n_latin_chars + n_deva
    deva_ratio = (n_deva / total_alpha) if total_alpha else 0.0
    has_deva = n_deva > 0
    has_latin = n_latin > 0

    hindi_hits = sum(1 for w in latin if w in HINGLISH_KEYWORDS)
    kw_ratio = (hindi_hits / n_latin) if n_latin else 0.0
    tech_present = any(w in _TECH_HINTS for w in latin)
    avg_len = (n_latin_chars / n_latin) if n_latin else 0.0

    f_deva = min(1.0, deva_ratio * 4.0)
    f_kw = min(1.0, kw_ratio / 0.25)
    f_kwc = min(1.0, hindi_hits / 3.0)
    f_mixed = 1.0 if (has_deva and has_latin) else 0.0
    f_tech = 1.0 if (tech_present and (hindi_hits > 0 or has_deva)) else 0.0
    f_len = max(0.0, min(1.0, (4.5 - avg_len) / 2.0)) if n_latin else 0.0
    if language and language != "en":
        f_lang = 1.0
    elif language_probability < 0.85:
        f_lang = 0.5
    else:
        f_lang = 0.0
    f_conf = max(0.0, min(1.0, (-0.55 - avg_logprob) / 0.5))

    score = (0.35 * f_deva + 0.25 * f_kw + 0.20 * f_kwc + 0.15 * f_mixed
             + 0.08 * f_tech + 0.04 * f_len + 0.10 * f_lang + 0.08 * f_conf)
    return max(0.0, min(1.0, score))


def should_escalate(
    language: str,
    language_probability: float,
    avg_logprob: float,
    compression_ratio: float,
    text: str,
    no_speech_prob: float = 0.0,
    words: Optional[list] = None,
    temperature: float = 0.0,
    all_language_probs: Optional[list] = None,
) -> bool:
    """Decide whether a clip needs the stronger Hindi-capable path.

    Recall-biased UNION of cheap signals (any one escalates). Thresholds are tightened
    vs. the original router to push recall up. The last five parameters are optional
    richer faster-whisper signals — neutral by default so the existing 5-arg call site
    keeps working; they activate automatically if the orchestrator ever passes them.
    """
    if not (text or "").strip():
        return True                                    # blank draft → verify
    if language and language != "en":
        return True                                    # any non-English top language
    if contains_devanagari(text):
        return True                                    # explicit code-switch
    if language_probability < 0.92:                    # was 0.85 — more recall
        return True
    if avg_logprob <= -0.45:                           # was -0.55 — more recall
        return True
    if compression_ratio >= 2.2:                       # was 2.4 — more recall
        return True
    if no_speech_prob >= 0.6:                           # noisy → hallucination risk
        return True
    if temperature and temperature >= 0.4:              # Whisper had to fall back
        return True
    if all_language_probs:                              # Indic mass even if top is English
        try:
            indic = sum(float(p) for lang, p in all_language_probs if lang in _INDIC_LANGS)
            if indic >= 0.10:
                return True
        except Exception:
            pass
    if words:                                           # any very-low-confidence word
        try:
            probs = [float(getattr(w, "probability", 1.0)) for w in words]
            if probs and min(probs) <= 0.35:
                return True
        except Exception:
            pass
    if detect_hinglish_score(text, language, language_probability, avg_logprob) >= HINGLISH_SCORE_THRESHOLD:
        return True                                    # romanized Hinglish (the key catch)
    return False


def route(
    text: str,
    language: str = "",
    language_probability: float = 1.0,
    avg_logprob: float = 0.0,
    compression_ratio: float = 0.0,
    no_speech_prob: float = 0.0,
    words: Optional[list] = None,
    temperature: float = 0.0,
    all_language_probs: Optional[list] = None,
) -> tuple[str, float]:
    """Final routing decision: returns ("HINGLISH"|"ENGLISH", confidence in [0,1]).

    Confidence is the Hinglish-likelihood score, raised to near-certainty when a hard
    signal fires (Devanagari / non-English language / blank draft)."""
    esc = should_escalate(language, language_probability, avg_logprob, compression_ratio,
                          text, no_speech_prob, words, temperature, all_language_probs)
    score = detect_hinglish_score(text, language, language_probability, avg_logprob)
    if contains_devanagari(text):
        score = max(score, 0.97)
    elif language and language != "en":
        score = max(score, 0.90)
    if not (text or "").strip():
        score = max(score, 0.80)

    if esc:
        return ("HINGLISH", round(max(score, 0.51), 3))
    return ("ENGLISH", round(1.0 - score, 3))


# =============================================================================
# PART 6 — Hinglish model loader (Qwen3-ASR 0.6B; GPU vLLM, transformers fallback)
# =============================================================================
class _HinglishHandle:
    """Wraps whichever Hinglish-capable engine actually loaded."""

    def __init__(self, kind: str, obj: Any, model_id: str) -> None:
        self.kind = kind          # "qwen" (transformers pipeline) | "fw" (faster-whisper)
        self.obj = obj
        self.model_id = model_id


def _model_footprint_mb(model: Any):
    """In-memory parameter footprint (MB) of the loaded model, best effort."""
    try:
        for sub in (getattr(model, "LLM", None), getattr(model, "model", None), model):
            if sub is not None and hasattr(sub, "parameters"):
                tot = sum(p.numel() * p.element_size() for p in sub.parameters())
                if tot:
                    return round(tot / 1e6)
    except Exception:
        pass
    return None


def _load_qwen_hinglish(meta: Optional[dict] = None) -> Optional[_HinglishHandle]:
    """Load the code-switch Qwen3-ASR model via the official ``qwen-asr`` package
    (Apache-2.0). Backend by device: CUDA→vLLM (bf16) → transformers (bf16/FA2);
    Apple→transformers MPS (fp16/sdpa); CPU→transformers (fp32/sdpa, fidelity-first).
    Local cache first; downloads ONCE when ``_downloads_allowed`` then loads offline
    forever. Returns ``None`` on failure."""
    global _LAST_HINGLISH_ERROR
    try:
        cached = False
        local_path = None
        try:
            from huggingface_hub import snapshot_download
            # capture the on-disk snapshot dir → load from the PATH (not the repo id) so the
            # offline scored run never makes an HF API call (which the network guard blocks).
            local_path = snapshot_download(HINGLISH_QWEN_NAME, local_files_only=True)
            cached = True
        except Exception:
            cached = False
        if not cached and not _downloads_allowed():
            _log("Qwen3-ASR not cached and downloads not allowed — skipping")
            return None

        from qwen_asr import Qwen3ASRModel  # local import: heavy deps loaded only when needed

        local_only = cached  # cached → offline (local_files_only); cold → one-time download
        model_ref = local_path if (cached and local_path) else HINGLISH_QWEN_NAME  # path skips API
        _log(f"loading Qwen3-ASR '{HINGLISH_QWEN_NAME}' (cached={cached}, local_files_only={local_only})")
        # NOTE: do NOT force HF_HUB_OFFLINE=1 — qwen-asr makes an API call that raises
        # OfflineModeIsEnabled under forced offline. local_files_only=True is the correct
        # offline switch: it uses the cache and skips the network.
        import torch
        on_cuda = _cuda_available()
        on_mps = _mps_available()      # Apple-silicon Metal (M1/M2/M3) — the live-track box
        backend_pref = os.environ.get("STT_QWEN_BACKEND", "auto")  # auto | vllm | transformers

        # ---- FASTEST: vLLM backend (CUDA only). Same model + same .transcribe() API. ----
        if on_cuda and backend_pref in ("auto", "vllm"):
            try:
                # Triton attention works across GPUs (incl. pre-Ampere T4); vLLM's default
                # FlashInfer needs a JIT build that fails on some boxes (ld: -lcuda). Overridable.
                os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
                import vllm  # noqa: F401
                t0 = time.time()
                vmodel = Qwen3ASRModel.LLM(
                    model_ref,
                    gpu_memory_utilization=float(os.environ.get("STT_GPU_MEM_UTIL", "0.85")),
                    max_inference_batch_size=32,
                    max_new_tokens=QWEN_MAX_NEW_TOKENS,
                )
                if meta is not None:
                    meta.update(model=HINGLISH_QWEN_NAME, kind="qwen-asr", backend="vllm",
                                device="cuda", precision="bf16", cached=cached, cache=_hf_cache_dir(),
                                load_ms=round((time.time() - t0) * 1000), quantization="off(vllm-bf16)",
                                max_new_tokens=QWEN_MAX_NEW_TOKENS, num_beams=1, do_sample=False)
                _LAST_HINGLISH_ERROR = ""
                _log("Qwen3-ASR LOADED via vLLM (bf16, GPU)")
                return _HinglishHandle("qwen", vmodel, HINGLISH_QWEN_ID)
            except Exception as e:  # noqa: BLE001 — fall back to transformers backend
                _log(f"vLLM unavailable, using transformers backend: {type(e).__name__}: {e}")

        # ---- transformers backend ----
        # CUDA: device_map + FA2/sdpa (validated on the T4). Apple MPS / CPU: load on CPU with
        # NO device_map and NO forced attn_implementation — the custom qwen3_asr arch + accelerate
        # can reject device_map="mps" and a forced sdpa kernel (this is the path that "doesn't run
        # on the box"). After loading we MOVE the module to the Apple GPU via .to("mps"); ANY
        # failure there falls back to CPU, so the model always loads and runs.
        load_kwargs: dict = {"max_new_tokens": QWEN_MAX_NEW_TOKENS}
        if local_only:
            load_kwargs["local_files_only"] = True
        if on_cuda:
            load_kwargs.update(device_map="cuda:0", dtype=_gpu_dtype())  # bf16 on Ampere+, else fp16
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                load_kwargs["attn_implementation"] = "sdpa"
        else:
            load_kwargs["dtype"] = torch.float32   # safe minimal CPU load; cast on the MPS move

        t0 = time.time()
        try:                                      # honor RULE-3 local_files_only when supported
            model = Qwen3ASRModel.from_pretrained(model_ref, **load_kwargs)
        except TypeError:
            load_kwargs.pop("local_files_only", None)
            model = Qwen3ASRModel.from_pretrained(model_ref, **load_kwargs)

        device = "cuda" if on_cuda else "cpu"
        if on_mps:                                # opportunistic Apple-GPU move; CPU on any failure
            try:
                inner = getattr(model, "model", None)
                if inner is not None:
                    inner.to("mps", dtype=torch.float16)
                    try:
                        model.device = next(inner.parameters()).device
                    except Exception:
                        pass
                    try:
                        model.dtype = torch.float16
                    except Exception:
                        pass
                    device = "mps"
                    _log("Qwen3-ASR moved to Apple MPS (fp16)")
            except Exception as e:  # noqa: BLE001 — MPS unsupported/unavailable → stay on CPU
                _log(f"MPS move failed ({type(e).__name__}: {e}); running Qwen on CPU")
                device = "cpu"
        load_ms = round((time.time() - t0) * 1000)
        # anti-runaway: stop degenerate repetition loops that otherwise generate all the way
        # to max_new_tokens (~80s on the T4 transformers backend). Mild, quality-safe.
        try:
            _gc = getattr(getattr(model, "model", None), "generation_config", None)
            if _gc is not None:
                _gc.no_repeat_ngram_size = 3
                _gc.repetition_penalty = 1.3   # stronger: kills residual repetition loops
        except Exception:
            pass
        precision = ("bf16" if (on_cuda and _gpu_dtype().__str__().endswith("bfloat16"))
                     else "fp16" if device in ("cuda", "mps") else "fp32")

        if meta is not None:
            meta.update(model=HINGLISH_QWEN_NAME, kind="qwen-asr", backend="transformers",
                        device=device, precision=precision,
                        local_files_only=local_only, cached=cached, cache=_hf_cache_dir(),
                        load_ms=load_ms, quantization="off(" + precision + ")",
                        footprint_mb=_model_footprint_mb(model),
                        max_new_tokens=QWEN_MAX_NEW_TOKENS, num_beams=1, do_sample=False)
        _LAST_HINGLISH_ERROR = ""
        _log(f"Qwen3-ASR LOADED (cached={cached}, device={device}/{precision}, {load_ms}ms)")
        return _HinglishHandle("qwen", model, HINGLISH_QWEN_ID)
    except Exception as e:
        _LAST_HINGLISH_ERROR = f"qwen: {type(e).__name__}: {e}"
        _log(f"Qwen3-ASR load failed — {_LAST_HINGLISH_ERROR}")
        return None


def _load_whisper_hinglish(meta: Optional[dict] = None) -> Optional[_HinglishHandle]:
    """Load the active Hinglish recognizer: a Whisper-large-v3 finetune
    (``Oriserve/Whisper-Hindi2Hinglish-*``, Apache-2.0) via the STANDARD transformers ASR
    pipeline. Standard architecture → loads on CUDA / Apple MPS / CPU with no custom code,
    which is why it runs on the M1 box where the custom qwen3_asr arch did not. Local cache
    first; downloads ONCE when allowed then loads offline. Returns ``None`` on failure."""
    global _LAST_HINGLISH_ERROR
    try:
        import torch
        from transformers import pipeline
        import numpy as _np

        def _resolve(model_name):
            # on-disk snapshot path (offline-safe) if cached, else the repo id (downloads once)
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(model_name, local_files_only=True), True
            except Exception:
                return model_name, False

        def _build(model_ref, dev, dt, offline):
            # Match the reference finalizer EXACTLY: pipeline(model, device, torch_dtype). Use
            # float32 on MPS (fp16-on-Metal is unreliable and silently drops to CPU ≈ 15s/decode
            # = "far too slow"); fp16 only on CUDA.
            mk = dict(use_safetensors=True, low_cpu_mem_usage=True,
                      **({"local_files_only": True} if offline else {}))
            try:
                return pipeline("automatic-speech-recognition", model=model_ref, device=dev,
                                torch_dtype=dt, chunk_length_s=30, model_kwargs=mk,
                                generate_kwargs={"task": "transcribe", "language": "en"})
            except TypeError:                 # older transformers: dtype kwarg name
                return pipeline("automatic-speech-recognition", model=model_ref, device=dev,
                                dtype=dt, chunk_length_s=30, model_kwargs=mk,
                                generate_kwargs={"task": "transcribe", "language": "en"})

        name = HINGLISH_WHISPER_NAME
        ref, cached = _resolve(name)
        if not cached and not _downloads_allowed():
            _log("Whisper-Hinglish not cached and downloads not allowed — skipping")
            return None

        on_cuda, on_mps = _cuda_available(), _mps_available()
        device = "cuda:0" if on_cuda else ("mps" if on_mps else "cpu")
        dtype = torch.float16 if on_cuda else torch.float32
        t0 = time.time()

        pipe = None
        # 1) Try the accelerator with the default (Apex) model. Wrap the WHOLE attempt —
        #    pipeline(device="mps") allocates Metal memory at BUILD time, so a broken GPU
        #    (e.g. a CI VM) raises here, not just during inference. Any failure → CPU.
        if device != "cpu":
            try:
                pipe = _build(ref, device, dtype, cached)
                pipe(_np.zeros(1600, dtype=_np.float32))   # force a real decode to confirm it works
            except Exception as e:            # noqa: BLE001
                _log(f"{device} unusable ({type(e).__name__}: {e}); using CPU")
                device, dtype, pipe = "cpu", torch.float32, None

        # 2) CPU path: a heavy model on CPU is ~15s/decode, so use the fast Swift variant
        #    (whisper-base, ~1-2s) unless the user pinned a model. Build exactly one model.
        if pipe is None:
            device, dtype = "cpu", torch.float32
            if not on_cuda and "STT_HINGLISH_MODEL" not in os.environ and "Swift" not in name:
                sname = "Oriserve/Whisper-Hindi2Hinglish-Swift"
                sref, scached = _resolve(sname)
                if scached or _downloads_allowed():
                    name, ref, cached = sname, sref, scached
                    _log("CPU-only box — using fast Swift variant for survivable latency")
            pipe = _build(ref, "cpu", torch.float32, cached)

        load_ms = round((time.time() - t0) * 1000)
        if meta is not None:
            meta.update(model=name, kind="whisper-hinglish", backend="transformers",
                        device=("cuda" if on_cuda else device),
                        precision=("fp16" if (on_cuda and device != "cpu") else "fp32"),
                        local_files_only=cached, cached=cached, cache=_hf_cache_dir(),
                        load_ms=load_ms, num_beams=1, do_sample=False)
        _LAST_HINGLISH_ERROR = ""
        _log(f"Whisper-Hinglish LOADED '{name}' (device={device}, {load_ms}ms)")
        return _HinglishHandle("whisper", pipe, HINGLISH_WHISPER_ID)
    except Exception as e:
        _LAST_HINGLISH_ERROR = f"whisper: {type(e).__name__}: {e}"
        _log(f"Whisper-Hinglish load failed — {_LAST_HINGLISH_ERROR}")
        return None


def get_hinglish_model() -> Optional[_HinglishHandle]:
    """Return the resident Hinglish recognizer: a Whisper-large-v3 finetune
    (``Oriserve/Whisper-Hindi2Hinglish-*``, Apache-2.0) via the standard transformers ASR
    pipeline. Prefers an accelerator (CUDA / Apple MPS) and falls back to CPU. Loaded once,
    cached at module scope. ``None`` on failure — the final then keeps the fast draft."""
    global HINGLISH_MODEL, _HINGLISH_TRIED, _LAST_HINGLISH_ERROR
    if HINGLISH_MODEL is not None:
        return HINGLISH_MODEL
    if _HINGLISH_TRIED:        # already failed once — don't retry the (possibly slow) load
        return None
    _HINGLISH_TRIED = True

    # No accelerator AND CPU explicitly disabled → skip (fast-path-only). Default: run on CPU.
    # Scoring on the M1 uses MPS, so this only affects GPU-less boxes (local testing).
    if not _gpu_available() and not _cpu_qwen_allowed():
        _LAST_HINGLISH_ERROR = "skipped: no GPU and STT_DISABLE_CPU_QWEN=1 (fast-path used)"
        _log(_LAST_HINGLISH_ERROR)
        return None
    if not _gpu_available():
        _log("no accelerator — loading Whisper-Hinglish on CPU (slower final)")

    # No one-shot cap when a download is expected; a generous cap when offline guards a
    # partial-cache hang (large-v3 weights load in a few seconds from a complete cache).
    to = None if _downloads_allowed() else 180.0
    try:
        HINGLISH_MODEL = _call_with_timeout(lambda: _load_whisper_hinglish(_HINGLISH_META), to)
    except Exception as e:
        _LAST_HINGLISH_ERROR = f"whisper: {type(e).__name__}: {e}"
        _log(f"Whisper-Hinglish load aborted — {_LAST_HINGLISH_ERROR}")
        HINGLISH_MODEL = None
    return HINGLISH_MODEL


def model_debug_status() -> dict:
    """Force-load both models (downloading once if allowed) and report status.
    Used by ``--debug``; harmless to call anywhere."""
    fast = get_fast_model()
    hing = get_hinglish_model()
    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "(unset)"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "(unset)"),
        "STT_OFFLINE": os.environ.get("STT_OFFLINE", "(unset)"),
        "network_blocked": _network_is_blocked(),
        "downloads_allowed": _downloads_allowed(),
        "hf_cache_dir": _hf_cache_dir(),
        "fast_loaded": fast is not None,
        "fast_model_id": FAST_MODEL_ID,
        "fast_meta": dict(_FAST_META),
        "fast_error": _LAST_FAST_ERROR,
        "hinglish_loaded": hing is not None,
        "hinglish_model_id": (hing.model_id if hing else None),
        "hinglish_meta": dict(_HINGLISH_META),
        "hinglish_error": _LAST_HINGLISH_ERROR,
    }


def debug_hinglish_model() -> dict:
    """Load the Hinglish model and report path / quantization / footprint / load time."""
    h = get_hinglish_model()
    meta = dict(_HINGLISH_META)
    bar = "=" * 56
    print(bar)
    print("HINGLISH MODEL DEBUG")
    print(bar)
    print(f"  loaded        : {h is not None}")
    print(f"  model_id      : {h.model_id if h else None}")
    print(f"  model         : {meta.get('model')}")
    print(f"  cache / path  : {meta.get('cache')}")
    print(f"  device        : {meta.get('device')}")
    print(f"  quantization  : {meta.get('quantization')}")
    print(f"  footprint     : {meta.get('footprint_mb')} MB")
    print(f"  load time     : {meta.get('load_ms')} ms")
    print(f"  local_only    : {meta.get('local_files_only')}  cached={meta.get('cached')}")
    if not h:
        print(f"  error         : {_LAST_HINGLISH_ERROR}")
    print(bar)
    return meta


# =============================================================================
# PART 7 — Hinglish transcription
# =============================================================================
def hinglish_transcribe(audio: AudioInput) -> dict[str, Any]:
    """Run the escalated Hindi-capable path. Keeps the code-switch faithful (no
    translation). Returns {"text", "time_ms", "model_id"}."""
    t0 = time.time()
    out: dict[str, Any] = {"text": "", "time_ms": 0, "model_id": ""}
    handle = get_hinglish_model()
    if handle is None:
        out["time_ms"] = round((time.time() - t0) * 1000)
        return out
    out["model_id"] = handle.model_id
    try:
        if handle.kind == "whisper":
            # transformers ASR pipeline: accepts a 16k float32 ndarray or a file path and
            # returns {"text": ...}. The finetune emits faithful Hinglish (Latin code-switch).
            res = handle.obj(audio)
            out["text"] = ((res.get("text", "") if isinstance(res, dict) else "") or "").strip()
        else:
            # legacy qwen-asr: file path or (np.ndarray, sample_rate) tuple
            audio_arg: Any = audio if isinstance(audio, str) else (audio, TARGET_SR)
            res = handle.obj.transcribe(audio_arg, language=None)
            item = res[0] if isinstance(res, list) and res else res
            out["text"] = (getattr(item, "text", "") or "").strip()
    except Exception:
        pass  # leave blank; the finalizer falls back to the fast draft
    out["time_ms"] = round((time.time() - t0) * 1000)
    return out


# =============================================================================
# PART 8 — finalizer
# =============================================================================
# Canonical casing for work terms (dictionary normalization — NOT phrase hacking).
_TECH_CANON = {
    "aws": "AWS", "gpt": "GPT", "docker": "Docker", "kubernetes": "Kubernetes",
    "jira": "Jira", "cursor": "Cursor", "prd": "PRD", "api": "API",
    "kafka": "Kafka", "redis": "Redis", "codex": "Codex", "kubectl": "kubectl",
    "p95": "p95", "p99": "p99",
}

# Spaced/spelled acronyms → joined form (targeted, to avoid clobbering real words).
_SPACED_ACRONYMS = [
    (re.compile(r"\ba[.\s]*w[.\s]*s\b", re.IGNORECASE), "AWS"),
    (re.compile(r"\bg[.\s]*p[.\s]*t\b", re.IGNORECASE), "GPT"),
    (re.compile(r"\ba[.\s]*p[.\s]*i\b", re.IGNORECASE), "API"),
    (re.compile(r"\bp[.\s]*r[.\s]*d\b", re.IGNORECASE), "PRD"),
]
_P9X = re.compile(r"(?i)\bp[\s\-]*((?:95|99))\b")          # "p 95" / "p-99" → p95 / p99
_TECH_WORD = re.compile(r"\b(" + "|".join(re.escape(k) for k in _TECH_CANON) + r")\b", re.IGNORECASE)


def _collapse_repeats(text: str) -> str:
    """Collapse degenerate n-gram loops ("hello hello hello hello" → "hello").
    Conservative: unigram runs collapse at >=3 reps; bi/tri-gram runs at >=2 reps."""
    toks = text.split()
    if not toks:
        return text
    # unigram runs first (so "hello hello hello hello" → "hello"), then phrase loops.
    for n, min_reps in ((1, 3), (2, 3), (3, 2)):
        if len(toks) < n * min_reps:
            continue
        out: list[str] = []
        i = 0
        while i < len(toks):
            gram = toks[i:i + n]
            if len(gram) == n:
                reps = 1
                j = i + n
                while toks[j:j + n] == gram:
                    reps += 1
                    j += n
                if reps >= min_reps:
                    out.extend(gram)   # keep a single copy
                    i = j
                    continue
            out.append(toks[i])
            i += 1
        toks = out
    return " ".join(toks)


def _normalize_terms(text: str) -> str:
    """Apply spaced-acronym joins, p95/p99 fixes, and canonical work-term casing.
    Latin-only edits — Devanagari is never touched."""
    for pat, repl in _SPACED_ACRONYMS:
        text = pat.sub(repl, text)
    text = _P9X.sub(lambda m: "p" + m.group(1), text)
    text = _TECH_WORD.sub(lambda m: _TECH_CANON[m.group(1).lower()], text)
    return text


def finalize_transcript(
    hinglish_text: str,
    fast_text: str,
    mode: str,
    escalated: bool,
) -> str:
    """Produce the final transcript.

    * Never translates Hindi (we only ever recase/normalize Latin tokens).
    * Preserves and normalizes work terms (AWS, GPT, Docker, p95, …).
    * Removes repetition loops.
    * Never returns blank when any candidate has content
      (fallback: hinglish → fast → best non-empty candidate).
    """
    h = (hinglish_text or "").strip()
    f = (fast_text or "").strip()

    # choose the base candidate
    if (escalated or mode in ("hinglish", "verbatim")) and h:
        base = h
    elif f:
        base = f
    else:
        base = h or f

    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        return ""

    if mode == "verbatim":
        # minimal normalization: keep words/script as-is, only kill loops + whitespace
        return _collapse_repeats(base).strip()

    base = _normalize_terms(base)
    base = _collapse_repeats(base)
    base = re.sub(r"\s+([,.!?;:])", r"\1", base)  # tidy spacing before punctuation
    base = re.sub(r"\s+", " ", base).strip()

    if not base:  # normalization should never empty it, but guard anyway
        return (h or f).strip()
    return base


# =============================================================================
# language guess
# =============================================================================
def _devanagari_only(text: str) -> list:
    return [t for t in (text or "").split() if contains_devanagari(t)]


def _apply_preserving_hindi(text: str, fn) -> str:
    """RULE 1 (absolute Hindi fidelity): run a normalize/repair step, but REJECT its
    output entirely if it changed, romanized, dropped, or re-spaced ANY Devanagari token.
    Devanagari tokens must survive verbatim through every post-ASR stage."""
    before = _devanagari_only(text)
    try:
        out = fn(text)
    except Exception:
        return text
    if not isinstance(out, str):
        return text
    if _devanagari_only(out) != before:      # a Hindi token was touched → discard the change
        return text
    return out


_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def _strip_arabic_if_hinglish(text: str, have_hinglish: bool) -> str:
    """TASK 5: the fast model sometimes emits Arabic/Urdu script (e.g. 'لیبر آفس') on
    Hindi audio. When a Hinglish (Devanagari) output exists, drop Arabic-script tokens.
    Never returns blank (keeps the original if stripping would empty it)."""
    if not have_hinglish or not text or not _ARABIC.search(text):
        return text
    kept = [tok for tok in text.split() if not _ARABIC.search(tok)]
    out = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return out if out else text


def _language_guess(final_text: str, fast_language: str) -> str:
    has_deva = contains_devanagari(final_text)
    has_latin = bool(re.search(r"[A-Za-z]", final_text or ""))
    if has_deva and has_latin:
        return "hinglish"
    if has_deva:
        return "hindi"
    if fast_language == "en" or has_latin:
        return "english"
    return fast_language or "unknown"


# =============================================================================
# PART 9 + PART 10 — orchestration + exact JSON contract
# =============================================================================
_VALID_MODES = ("auto", "fast", "hinglish", "verbatim")


def transcribe(wav_path: str, mode: str = "auto") -> dict:
    """Transcribe ``wav_path`` and return the required result contract.

    Modes:
      * ``auto``     — fast ASR → router → optional Hinglish escalation → finalizer.
      * ``fast``     — fast model only (lowest latency, no escalation).
      * ``hinglish`` — always run the Hinglish model (fast used only as a fallback).
      * ``verbatim`` — Hinglish model, minimal normalization (Devanagari preserved).
    """
    t0 = time.time()
    mode = mode if mode in _VALID_MODES else "auto"

    raw_candidates: list[dict[str, Any]] = []
    model_ids: list[str] = []
    fast_text = ""
    hinglish_text = ""
    fast_language = ""
    asr_ms = 0
    escalated = False

    try:
        # ---- preprocess ----
        audio = load_audio(wav_path, do_vad=True)
        asr_input: AudioInput = audio if audio is not None else wav_path

        # ---- fast pass (skipped up-front for hinglish/verbatim; used as fallback there) ----
        run_fast_first = mode in ("auto", "fast")
        if run_fast_first:
            fr = fast_transcribe(asr_input)
            fast_text = fr["text"]
            fast_language = fr["language"]
            asr_ms += int(fr["time_ms"])
            if get_fast_model() is not None:
                model_ids.append(FAST_MODEL_ID)
            raw_candidates.append({
                "engine": FAST_MODEL_ID, "text": fast_text,
                "language": fast_language,
                "language_probability": round(float(fr["language_probability"]), 3),
                "avg_logprob": round(float(fr["avg_logprob"]), 3),
                "compression_ratio": round(float(fr["compression_ratio"]), 3),
                "no_speech_prob": round(float(fr["no_speech_prob"]), 3),
            })

            if mode == "auto":
                escalated = bool(ENABLE_ROUTER) and should_escalate(
                    fast_language, float(fr["language_probability"]),
                    float(fr["avg_logprob"]), float(fr["compression_ratio"]), fast_text,
                )

        # ---- Hinglish / escalated pass ----
        want_hinglish = (mode in ("hinglish", "verbatim")
                         or (mode == "auto" and escalated)) and bool(ENABLE_HINGLISH)
        if want_hinglish:
            hr = hinglish_transcribe(asr_input)
            hinglish_text = hr["text"]
            asr_ms += int(hr["time_ms"])
            if hr.get("model_id"):
                escalated = True
                if hr["model_id"] not in model_ids:
                    model_ids.append(hr["model_id"])
                raw_candidates.append({"engine": hr["model_id"], "text": hinglish_text})

        # ---- fallback fast pass for hinglish/verbatim if the heavy path gave nothing ----
        if mode in ("hinglish", "verbatim") and not hinglish_text.strip():
            fr = fast_transcribe(asr_input)
            fast_text = fr["text"]
            fast_language = fr["language"]
            asr_ms += int(fr["time_ms"])
            if get_fast_model() is not None and FAST_MODEL_ID not in model_ids:
                model_ids.append(FAST_MODEL_ID)
            raw_candidates.append({
                "engine": FAST_MODEL_ID, "text": fast_text, "language": fast_language,
            })

        # ---- finalize → vocab/repair polish ----
        p0 = time.time()
        final_text = finalize_transcript(hinglish_text, fast_text, mode, escalated)

        # RULE 1: vocab/repair may only touch Latin (tech/number/spacing/caps); the guard
        # discards any change that would alter a Devanagari token.
        if ENABLE_VOCAB and _vocab is not None:
            final_text = _apply_preserving_hindi(final_text, _vocab.normalize_tech_words)
        if ENABLE_REPAIR and _vocab is not None:
            final_text = _apply_preserving_hindi(final_text, _vocab.repair_common_asr_errors)
        # Drop Arabic/Urdu-script leakage from the fast model when a Hinglish (Devanagari)
        # transcript exists.
        final_text = _strip_arabic_if_hinglish(final_text, bool(hinglish_text.strip()))
        post_ms = round((time.time() - p0) * 1000)

        language_guess = _language_guess(final_text, fast_language)
        if mode == "auto":
            # report "hinglish" only when the escalated path actually produced output
            ran_hinglish = any(c.get("engine") == HINGLISH_QWEN_ID
                               and (c.get("text") or "").strip() for c in raw_candidates)
            mode_used = "hinglish" if (escalated and ran_hinglish) else "fast"
        else:
            mode_used = mode

    except Exception:
        # absolute safety net — emit a valid (blank) contract rather than crashing
        final_text = (hinglish_text or fast_text or "").strip()
        language_guess = _language_guess(final_text, fast_language)
        mode_used = mode
        post_ms = 0
        if not raw_candidates:
            raw_candidates = [{"engine": "none", "text": final_text}]

    total_ms = round((time.time() - t0) * 1000)
    return {
        "text": final_text,
        "mode_used": mode_used,
        "language_guess": language_guess,
        "timings_ms": {"total": total_ms, "asr": int(asr_ms), "postprocess": int(post_ms)},
        "raw_candidates": raw_candidates,
        "model_ids": model_ids,
        "local_only": True,
    }


def _print_debug_status() -> None:
    st = model_debug_status()
    bar = "=" * 56
    print(bar)
    print("MODEL DEBUG STATUS")
    print(bar)
    print(f"HF OFFLINE STATUS        : HF_HUB_OFFLINE={st['HF_HUB_OFFLINE']}")
    print(f"TRANSFORMERS OFFLINE     : {st['TRANSFORMERS_OFFLINE']}")
    print(f"STT_OFFLINE              : {st['STT_OFFLINE']}")
    print(f"NETWORK BLOCKED          : {st['network_blocked']}")
    print(f"DOWNLOADS ALLOWED        : {st['downloads_allowed']}")
    print(f"CACHE STATUS (dir)       : {st['hf_cache_dir']}")
    print("-" * 56)
    print(f"FAST MODEL STATUS        : {'LOADED' if st['fast_loaded'] else 'NOT LOADED'}")
    print(f"  model_id               : {st['fast_model_id']}")
    print(f"  meta                   : {st['fast_meta'] or '{}'}")
    if not st["fast_loaded"]:
        print(f"  error                  : {st['fast_error']}")
    print("-" * 56)
    print(f"HINGLISH MODEL STATUS    : {'LOADED' if st['hinglish_loaded'] else 'NOT LOADED'}")
    print(f"  model_id               : {st['hinglish_model_id']}")
    print(f"  meta                   : {st['hinglish_meta'] or '{}'}")
    if not st["hinglish_loaded"]:
        print(f"  error                  : {st['hinglish_error']}")
    print(bar)
    if st["fast_loaded"] or st["hinglish_loaded"]:
        print("Models are cached locally; subsequent runs load from the local cache (offline).")
    else:
        print("NO MODEL LOADED.")
        if "ModuleNotFoundError" in (st["fast_error"] or ""):
            print("  → The ASR package isn't installed. Run:  pip install -r requirements.txt")
        print("  → Then warm the cache once with the network available:")
        print("       HF_HUB_OFFLINE=0 python -m solution.transcribe --debug")


def main() -> None:
    ap = argparse.ArgumentParser(description="builderr local dictation engine")
    ap.add_argument("--input")
    ap.add_argument("--mode", default="auto", choices=list(_VALID_MODES))
    ap.add_argument("--output")
    ap.add_argument("--debug", action="store_true",
                    help="print FAST/HINGLISH model + cache + offline status and exit "
                         "(downloads once if absent and the network is available)")
    args = ap.parse_args()

    if args.debug:
        global VERBOSE
        VERBOSE = True
        _print_debug_status()
        return

    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --debug)")

    try:
        result = transcribe(args.input, args.mode)
    except Exception as e:  # noqa: BLE001 — CLI must never crash mid-pipe
        result = {
            "text": "", "mode_used": args.mode, "language_guess": "unknown",
            "timings_ms": {"total": 0, "asr": 0, "postprocess": 0},
            "raw_candidates": [{"engine": "none", "text": "", "note": type(e).__name__}],
            "model_ids": [], "local_only": True,
        }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        print(f"failed to write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, "
          f"mode_used={result['mode_used']}, local_only={result['local_only']})")


if __name__ == "__main__":
    main()
