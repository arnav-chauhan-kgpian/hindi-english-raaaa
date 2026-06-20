"""builderr · benchmarking harness (read-only analysis — does NOT modify the engine).

Benchmarks the two recognizers used by solution/transcribe.py, independently, on the
local English + Hinglish clips, then analyzes the router and emits a data-backed
recommendation. It imports the existing engine and scorer — it changes nothing.

    python benchmark.py                 # uses samples/ + data/dev/ if present
    python benchmark.py --limit 8       # quick subsample
    python benchmark.py --modes fast    # benchmark only the fast model

What it measures, per model, split by English vs Hinglish:
    WER (scorecard.wer) · avg/p50/p95 decode latency · model load time · peak RSS.
Plus router stats: escalation rate, false positives (English escalated),
false negatives (Hinglish missed).

Note on WER: scorecard.wer normalizes ASCII-only (the harness's deterministic proxy),
so on code-switch clips it scores only the embedded Latin tokens. Treat Hinglish WER
here as the proxy the leaderboard uses, not as a full faithfulness measure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scorecard import judge_meaning, percentile, wer  # noqa: E402
import solution.transcribe as T  # noqa: E402


# --------------------------------------------------------------------------- #
# memory / timing helpers
# --------------------------------------------------------------------------- #
def get_rss_mb() -> Optional[float]:
    """Current process RSS in MB (psutil), or peak via resource as a fallback."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        try:
            import resource

            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (r / 1e6) if sys.platform == "darwin" else (r / 1024.0)  # mac=bytes, linux=KB
        except Exception:
            return None


def fmt_ms(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0f} ms"


def fmt_wer(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def fmt_mb(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0f} MB"


def mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


# --------------------------------------------------------------------------- #
# clip loading + labeling
# --------------------------------------------------------------------------- #
def _label(clip: dict) -> str:
    """Classify a clip into english / hinglish / hindi / other from manifest fields."""
    lang = (clip.get("language") or "").lower()
    cat = (clip.get("category") or "").lower()
    if lang in ("hi-en", "hinglish") or "hinglish" in cat or "code" in cat:
        return "hinglish"
    if lang in ("english", "en") or "english" in cat or "fleurs_en" in cat or cat.startswith("youtube"):
        return "english"
    if lang in ("hindi", "hi") or "hindi" in cat:
        return "hindi"
    return "other"


def collect_clips(limit: Optional[int]) -> list[dict]:
    """Gather (clip_id, wav, gold, must_have, label) from samples/ and data/dev/ if present."""
    clips: list[dict] = []
    seen: set[str] = set()

    # 1) samples/manifest.json — ships audio paths directly
    sm = os.path.join(HERE, "samples", "manifest.json")
    if os.path.exists(sm):
        try:
            for c in json.load(open(sm, encoding="utf-8")):
                wav = os.path.join(HERE, c.get("audio", ""))
                if c["clip_id"] not in seen and os.path.exists(wav):
                    clips.append({"clip_id": c["clip_id"], "wav": wav, "gold": c.get("gold", ""),
                                  "must_have": c.get("must_have", []), "label": _label(c),
                                  "src": "samples"})
                    seen.add(c["clip_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  (warn: could not read samples manifest: {type(e).__name__})")

    # 2) data/dev/manifest.json — audio fetched into data/dev/audio/<id>.wav
    dm = os.path.join(HERE, "data", "dev", "manifest.json")
    if os.path.exists(dm):
        try:
            for c in json.load(open(dm, encoding="utf-8")):
                wav = os.path.join(HERE, "data", "dev", "audio", c["clip_id"] + ".wav")
                if c["clip_id"] not in seen and os.path.exists(wav):
                    clips.append({"clip_id": c["clip_id"], "wav": wav, "gold": c.get("gold", ""),
                                  "must_have": c.get("must_have", []), "label": _label(c),
                                  "src": "dev"})
                    seen.add(c["clip_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  (warn: could not read dev manifest: {type(e).__name__})")

    clips.sort(key=lambda x: x["clip_id"])
    if limit:
        clips = clips[:limit]
    return clips


# --------------------------------------------------------------------------- #
# per-model benchmark
# --------------------------------------------------------------------------- #
def bench_model(kind: str, clips: list[dict], audio_cache: dict[str, Any]) -> dict[str, Any]:
    """Benchmark one model ('fast' or 'hinglish') over all clips.

    Returns per-bucket WER + latency stats, load time, peak RSS, and (for 'fast')
    the per-clip router signals so the caller can analyze the router.
    """
    res: dict[str, Any] = {
        "available": False, "load_ms": None, "rss_mb": None, "peak_rss_mb": None,
        "buckets": {}, "signals": {},  # signals: clip_id -> dict (fast only)
    }

    rss_before = get_rss_mb()
    t0 = time.time()
    if kind == "fast":
        model = T.get_fast_model()
    else:
        model = T.get_hinglish_model()
    load_ms = (time.time() - t0) * 1000.0

    if model is None:
        return res  # unavailable — caller prints n/a and guidance
    res["available"] = True
    res["load_ms"] = load_ms
    rss_after = get_rss_mb()
    res["rss_mb"] = (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else rss_after

    # warm the hinglish model once (the fast model is warmed inside get_fast_model)
    if kind == "hinglish":
        try:
            import numpy as np

            _ = T.hinglish_transcribe(np.zeros(T.TARGET_SR // 2, dtype="float32"))
        except Exception:
            pass

    by_bucket: dict[str, dict[str, list[float]]] = {}
    peak_rss = res["rss_mb"]
    for c in clips:
        audio = audio_cache.get(c["clip_id"])
        asr_input = audio if audio is not None else c["wav"]
        try:
            t = time.time()
            if kind == "fast":
                out = T.fast_transcribe(asr_input)
                pred = out.get("text", "")
                res["signals"][c["clip_id"]] = out
            else:
                out = T.hinglish_transcribe(asr_input)
                pred = out.get("text", "")
            latency = (time.time() - t) * 1000.0
        except Exception:
            pred, latency = "", 0.0

        b = by_bucket.setdefault(c["label"], {"wer": [], "lat": []})
        try:
            b["wer"].append(wer(c["gold"], pred))
        except Exception:
            pass
        b["lat"].append(latency)

        cur = get_rss_mb()
        if cur is not None:
            peak_rss = cur if peak_rss is None else max(peak_rss, cur)

    res["peak_rss_mb"] = peak_rss
    for bucket, d in by_bucket.items():
        lat = d["lat"]
        res["buckets"][bucket] = {
            "n": len(lat),
            "wer": mean(d["wer"]),
            "avg_ms": mean(lat),
            "p50_ms": percentile(lat, 0.50) if lat else None,
            "p95_ms": percentile(lat, 0.95) if lat else None,
        }
    return res


# --------------------------------------------------------------------------- #
# router analysis (uses the fast model's per-clip signals)
# --------------------------------------------------------------------------- #
def analyze_router(clips: list[dict], signals: dict[str, dict]) -> dict[str, Any]:
    total = escalated = 0
    fp = fn = 0
    n_eng = n_hi = 0
    decisions: list[tuple[str, str, bool]] = []  # (clip_id, label, escalated)
    for c in clips:
        sig = signals.get(c["clip_id"])
        if sig is None:
            continue
        total += 1
        esc = T.should_escalate(
            sig.get("language", ""), float(sig.get("language_probability", 0.0) or 0.0),
            float(sig.get("avg_logprob", 0.0) or 0.0),
            float(sig.get("compression_ratio", 0.0) or 0.0), sig.get("text", ""),
        )
        if esc:
            escalated += 1
        decisions.append((c["clip_id"], c["label"], esc))
        if c["label"] == "english":
            n_eng += 1
            if esc:
                fp += 1
        elif c["label"] == "hinglish":
            n_hi += 1
            if not esc:
                fn += 1
    return {
        "total": total, "escalated": escalated,
        "esc_rate": (escalated / total) if total else None,
        "fp": fp, "n_eng": n_eng, "fp_rate": (fp / n_eng) if n_eng else None,
        "fn": fn, "n_hi": n_hi, "fn_rate": (fn / n_hi) if n_hi else None,
        "decisions": decisions,
    }


# --------------------------------------------------------------------------- #
# recommendation
# --------------------------------------------------------------------------- #
def recommend(fast: dict, hing: dict, router: dict) -> tuple[str, str, list[str]]:
    """Choose A/B/C/D from the measured numbers. Returns (letter, title, reasons)."""
    LAT_GATE = 3500.0  # Hinglish p95 budget from the brief; >5000 is a hard hang

    def bw(m: dict, bucket: str, key: str) -> Optional[float]:
        return (m.get("buckets", {}).get(bucket, {}) or {}).get(key)

    fe, fh = bw(fast, "english", "wer"), bw(fast, "hinglish", "wer")
    he, hh = bw(hing, "english", "wer"), bw(hing, "hinglish", "wer")
    f_p95 = bw(fast, "english", "p95_ms")
    h_p95 = bw(hing, "hinglish", "p95_ms") or bw(hing, "english", "p95_ms")
    fn_rate = router.get("fn_rate")
    fp_rate = router.get("fp_rate")

    reasons: list[str] = []

    if not fast.get("available") and not hing.get("available"):
        return ("—", "INSUFFICIENT DATA — no model loaded",
                ["Neither model is installed/cached in this environment.",
                 "Install faster-whisper (and the Hinglish model), warm once with HF_HUB_OFFLINE=0, then re-run."])
    if not hing.get("available"):
        reasons.append("Hinglish model unavailable — cannot compare; benchmark the fast path only.")
        return ("B", "Remove router (fast-only) — provisional", reasons)
    if not fast.get("available"):
        reasons.append("Fast model unavailable — router cannot run; only the Hinglish path is measurable.")
        return ("C", "Always use Hinglish model — provisional", reasons)

    # core deltas (guard Nones)
    helps_mix = (fh - hh) if (fh is not None and hh is not None) else None     # + = hinglish better on mix
    hurts_eng = (he - fe) if (he is not None and fe is not None) else None     # + = hinglish worse on English
    hing_fast_enough = (h_p95 is not None and h_p95 <= LAT_GATE)

    if helps_mix is not None:
        reasons.append(f"Hinglish model changes mix WER by {helps_mix:+.3f} vs fast "
                       f"({fmt_wer(fh)} → {fmt_wer(hh)}).")
    if hurts_eng is not None:
        reasons.append(f"On English the Hinglish model is {hurts_eng:+.3f} WER vs fast "
                       f"({fmt_wer(fe)} → {fmt_wer(he)}).")
    if h_p95 is not None:
        reasons.append(f"Hinglish p95 {fmt_ms(h_p95)} vs fast p95 {fmt_ms(f_p95)} "
                       f"(gate {LAT_GATE:.0f} ms).")
    if fn_rate is not None:
        reasons.append(f"Router false-negatives (missed Hinglish): {router['fn']}/{router['n_hi']} "
                       f"= {fn_rate:.0%}; false-positives (English escalated): "
                       f"{router['fp']}/{router['n_eng']} = {fp_rate:.0%}.")

    # decision rules (data-driven, prioritized)
    if helps_mix is not None and helps_mix < 0.03 and (hurts_eng is None or hurts_eng >= -0.01):
        reasons.append("→ The Hinglish model does not meaningfully beat the fast model on the mix, "
                       "so escalation buys little. Drop the router and run the fast model everywhere.")
        return ("B", "Remove router — fast model is sufficient", reasons)

    if helps_mix is not None and helps_mix >= 0.05 and hing_fast_enough and \
            (hurts_eng is None or hurts_eng <= 0.02):
        reasons.append("→ The Hinglish model clearly wins the mix, doesn't hurt English, and stays under "
                       "the latency gate. Simplest robust choice is to run it on every clip.")
        return ("C", "Always use the Hinglish model", reasons)

    if helps_mix is not None and helps_mix >= 0.05 and not hing_fast_enough and \
            (fn_rate is not None and fn_rate <= 0.15) and (fp_rate is not None and fp_rate <= 0.20):
        reasons.append("→ The Hinglish model wins the mix but is too slow to run on everything; the router "
                       "is accurate (low FN/FP), so keep escalating only when needed.")
        return ("A", "Keep current architecture (router)", reasons)

    if helps_mix is not None and helps_mix >= 0.05 and (fn_rate is not None and fn_rate > 0.15):
        if h_p95 is not None and f_p95 is not None and (h_p95 + f_p95) <= 5000.0 and not hing_fast_enough:
            reasons.append("→ The Hinglish model wins the mix but the router misses too many code-switch clips, "
                           "and always-Hinglish exceeds the gate. Run both and merge by language/confidence.")
            return ("D", "Ensemble both outputs", reasons)
        reasons.append("→ The Hinglish model wins the mix and the router misses code-switch clips; if latency "
                       "allows, run it on every clip rather than risk the misses.")
        return ("C", "Always use the Hinglish model", reasons)

    reasons.append("→ Mixed signal: the router is the safe default — it preserves fast-model latency on English "
                   "while still escalating the risky clips.")
    return ("A", "Keep current architecture (router)", reasons)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
BAR = "=" * 48


def _print_model_block(title: str, res: dict) -> None:
    print(BAR)
    print(title)
    if not res.get("available"):
        print("  UNAVAILABLE — model not installed/cached in this environment.")
        print("  (install the engine + warm the cache with HF_HUB_OFFLINE=0, then re-run)")
        print()
        return
    print(f"  load time: {fmt_ms(res['load_ms'])}   incremental RSS: {fmt_mb(res['rss_mb'])}   "
          f"peak RSS: {fmt_mb(res['peak_rss_mb'])}")
    for bucket in ("english", "hinglish"):
        b = res["buckets"].get(bucket)
        print()
        print(f"  {bucket.capitalize()}:")
        if not b or b["n"] == 0:
            print("    (no clips available)")
            continue
        print(f"    n:       {b['n']}")
        print(f"    WER:     {fmt_wer(b['wer'])}")
        print(f"    Latency: {fmt_ms(b['avg_ms'])}")
        print(f"    p50:     {fmt_ms(b['p50_ms'])}")
        print(f"    p95:     {fmt_ms(b['p95_ms'])}")
        print(f"    RAM:     {fmt_mb(res['peak_rss_mb'])}")
    print()


# --------------------------------------------------------------------------- #
# ablation: toggle pipeline stages in transcribe.py and measure end-to-end
# --------------------------------------------------------------------------- #
_ALL_FLAGS = ("ENABLE_ROUTER", "ENABLE_HINGLISH", "ENABLE_ENSEMBLE", "ENABLE_VOCAB", "ENABLE_REPAIR")

# (name, {ROUTER, HINGLISH, ENSEMBLE, VOCAB, REPAIR})  — incremental ladder
ABLATION_CONFIGS = [
    ("1. Whisper only",                        (False, False, False, False, False)),
    ("2. Whisper + Router",                    (True,  False, False, False, False)),
    ("3. Whisper + Router + Hinglish",         (True,  True,  False, False, False)),
    ("4. + Ensemble",                          (True,  True,  True,  False, False)),
    ("5. Full pipeline (+ Vocab + Repair)",    (True,  True,  True,  True,  True)),
]
# extra eval-only configs to attribute Vocab vs Repair separately (built on config 4)
_ATTRIB_CONFIGS = [
    ("4 + Vocab only",  (True, True, True, True,  False)),
    ("4 + Repair only", (True, True, True, False, True)),
]


def _set_flags(combo) -> dict:
    """Set transcribe.ENABLE_* from a 5-tuple; return the previous values."""
    prev = {f: getattr(T, f) for f in _ALL_FLAGS}
    for f, v in zip(_ALL_FLAGS, combo):
        setattr(T, f, v)
    return prev


def _restore_flags(prev: dict) -> None:
    for f, v in prev.items():
        setattr(T, f, v)


def _eval_config(combo, clips: list[dict]) -> dict:
    """Run transcribe() over all clips under one flag combo; aggregate metrics."""
    prev = _set_flags(combo)
    eng_wer: list[float] = []
    hi_wer: list[float] = []
    meaning: list[float] = []
    lat: list[float] = []
    try:
        for c in clips:
            t = time.time()
            try:
                r = T.transcribe(c["wav"], "auto")
                pred = r.get("text", "")
            except Exception:
                pred = ""
            lat.append((time.time() - t) * 1000.0)
            try:
                w = wer(c["gold"], pred)
                meaning.append(judge_meaning(c["gold"], pred))
            except Exception:
                w = 1.0
            if c["label"] == "english":
                eng_wer.append(w)
            elif c["label"] == "hinglish":
                hi_wer.append(w)
    finally:
        _restore_flags(prev)
    return {
        "eng_wer": mean(eng_wer), "hi_wer": mean(hi_wer), "meaning": mean(meaning),
        "avg_ms": mean(lat),
        "p50_ms": percentile(lat, 0.50) if lat else None,
        "p95_ms": percentile(lat, 0.95) if lat else None,
    }


def run_ablation(limit: Optional[int] = None) -> None:
    """Measure the 5-config ablation ladder end-to-end and print a comparison +
    biggest-gain attribution + a submit recommendation."""
    clips = collect_clips(limit)
    if not clips:
        print("No clips with audio present — cannot run ablation "
              "(samples/*.wav ship with the repo; data/dev needs fetch_audio.py).")
        return
    counts: dict[str, int] = {}
    for c in clips:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    print(f"Ablation over {len(clips)} clips: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # warm models once so load time doesn't pollute the first config's latency
    try:
        T.get_fast_model()
        T.get_hinglish_model()
    except Exception:
        pass

    results = [(name, _eval_config(combo, clips)) for name, combo in ABLATION_CONFIGS]

    print()
    print(BAR)
    header = f"{'CONFIG':38s} {'EngWER':>7} {'HiWER':>7} {'Mean':>6} {'Avg':>7} {'p50':>7} {'p95':>7}"
    print(header)
    print("-" * len(header))
    for name, m in results:
        print(f"{name:38s} {fmt_wer(m['eng_wer']):>7} {fmt_wer(m['hi_wer']):>7} "
              f"{fmt_wer(m['meaning']):>6} {fmt_ms(m['avg_ms']):>7} "
              f"{fmt_ms(m['p50_ms']):>7} {fmt_ms(m['p95_ms']):>7}")
    print(BAR)

    # ---- pick BEST CONFIG (highest meaning; tie → lower p95) ----
    def _key(item):
        m = item[1]
        return (-(m["meaning"] or 0.0), (m["p95_ms"] if m["p95_ms"] is not None else 9e9))
    best_name, best_m = sorted(results, key=_key)[0]
    print(f"\nBEST CONFIG: {best_name}")
    print(f"  meaning={fmt_wer(best_m['meaning'])}  HiWER={fmt_wer(best_m['hi_wer'])}  "
          f"EngWER={fmt_wer(best_m['eng_wer'])}  p95={fmt_ms(best_m['p95_ms'])}")

    # ---- biggest-gain attribution (delta in meaning between ladder steps) ----
    def meaning_of(name_prefix):
        for n, m in results:
            if n.startswith(name_prefix):
                return m["meaning"] or 0.0
        return 0.0

    m1, m2, m3, m4, m5 = (results[i][1]["meaning"] or 0.0 for i in range(5))
    # attribute vocab vs repair via the extra configs
    attrib = {name: _eval_config(combo, clips)["meaning"] or 0.0 for name, combo in _ATTRIB_CONFIGS}
    gains = {
        "Router":     m2 - m1,
        "Hinglish":   m3 - m2,
        "Ensemble":   m4 - m3,
        "Vocabulary": attrib.get("4 + Vocab only", m4) - m4,
        "Repair":     attrib.get("4 + Repair only", m4) - m4,
    }
    print("\nMODULE GAIN (Δ meaning vs the previous stage):")
    for k, v in gains.items():
        print(f"  {k:12s} {v:+.3f}")
    biggest = max(gains, key=lambda k: gains[k])
    print(f"  → biggest gain: {biggest} ({gains[biggest]:+.3f})")

    # ---- submit recommendation (data-driven A/B/C) ----
    print(f"\n{BAR}\nRECOMMENDATION — what to submit")
    full = results[4][1]
    router_hing = results[2][1]
    whisper_only = results[0][1]
    if (full["meaning"] or 0) <= 1e-6:
        print("  INSUFFICIENT DATA — no ASR model loaded in this environment. Install + warm")
        print("  the models (HF_HUB_OFFLINE=0) and re-run `python benchmark.py --ablation`.")
        print(BAR)
        return
    gain_full_over_rh = (full["meaning"] or 0) - (router_hing["meaning"] or 0)
    gain_rh_over_w = (router_hing["meaning"] or 0) - (whisper_only["meaning"] or 0)
    p95_ok = full["p95_ms"] is not None and full["p95_ms"] <= 5000
    if gain_rh_over_w < 0.03:
        print("  A. Whisper only — Hinglish/router add < 0.03 meaning here; ship the simplest path.")
    elif gain_full_over_rh < 0.02:
        print("  B. Router + Hinglish — the big win is the Hinglish path; ensemble+vocab+repair add "
              f"only {gain_full_over_rh:+.3f}. Ship B for less complexity/latency.")
    else:
        flag = "" if p95_ok else "  (warning: p95 exceeds the 5s gate — verify latency)"
        print(f"  C. Full Pipeline — it scores highest (meaning {fmt_wer(full['meaning'])}, "
              f"+{gain_full_over_rh:.3f} over Router+Hinglish).{flag}")
    print(BAR)
    print("  (WER/meaning use scorecard's ASCII-only proxy; the hidden set uses an LLM judge. "
          "Dev numbers are directional.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="benchmark the fast + Hinglish recognizers")
    ap.add_argument("--limit", type=int, default=None, help="benchmark only the first N clips")
    ap.add_argument("--modes", default="fast,hinglish", help="comma list: fast,hinglish")
    ap.add_argument("--ablation", action="store_true", help="run the pipeline ablation ladder")
    args = ap.parse_args()

    if args.ablation:
        run_ablation(args.limit)
        return
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    clips = collect_clips(args.limit)
    if not clips:
        print("No clips found with audio present.")
        print("  • samples/*.wav ship with the repo (English + Hinglish).")
        print("  • data/dev/audio/<id>.wav appears after scripts/fetch_audio.py runs.")
        sys.exit(0)

    counts: dict[str, int] = {}
    for c in clips:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    print(f"Clips: {len(clips)}  by label: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if counts.get("hindi"):
        print(f"  (note: {counts['hindi']} pure-Hindi clip(s) excluded from the English/Hinglish buckets; "
              "ASCII-only WER is unreliable there)")
    print("Preloading audio (16k mono) ...")
    audio_cache: dict[str, Any] = {}
    for c in clips:
        try:
            audio_cache[c["clip_id"]] = T.load_audio(c["wav"], do_vad=True)
        except Exception:
            audio_cache[c["clip_id"]] = None
    print()

    fast_res: dict[str, Any] = {"available": False, "buckets": {}, "signals": {}}
    hing_res: dict[str, Any] = {"available": False, "buckets": {}, "signals": {}}

    if "fast" in modes:
        print("Benchmarking FAST model (faster-whisper small) ...")
        fast_res = bench_model("fast", clips, audio_cache)
    if "hinglish" in modes:
        print("Benchmarking HINGLISH model (qwen3-asr-0.6b-hinglish / fallback) ...")
        hing_res = bench_model("hinglish", clips, audio_cache)

    router = analyze_router(clips, fast_res.get("signals", {})) if fast_res.get("signals") else {
        "total": 0, "escalated": 0, "esc_rate": None, "fp": 0, "n_eng": 0, "fp_rate": None,
        "fn": 0, "n_hi": 0, "fn_rate": None, "decisions": [],
    }

    print()
    _print_model_block("FAST MODEL  (faster-whisper small)", fast_res)
    _print_model_block("HINGLISH MODEL  (qwen3-asr-0.6b-hinglish)", hing_res)

    print(BAR)
    print("ROUTER ANALYSIS")
    print(f"  Escalation Rate: {router['escalated']}/{router['total']}"
          + (f" = {router['esc_rate']:.0%}" if router["esc_rate"] is not None else " = n/a"))
    print(f"  False Positives (English escalated): {router['fp']}/{router['n_eng']}"
          + (f" = {router['fp_rate']:.0%}" if router["fp_rate"] is not None else " = n/a"))
    print(f"  False Negatives (Hinglish missed):   {router['fn']}/{router['n_hi']}"
          + (f" = {router['fn_rate']:.0%}" if router["fn_rate"] is not None else " = n/a"))
    print()

    letter, title, reasons = recommend(fast_res, hing_res, router)
    print(BAR)
    print("RECOMMENDATION")
    print(f"  {letter}. {title}")
    print()
    print("  Options: A=keep router  B=remove router (fast only)  "
          "C=always Hinglish  D=ensemble both")
    print()
    for r in reasons:
        print(f"  - {r}")
    print(BAR)
    print("  (WER uses scorecard's ASCII-only proxy; on the hidden set an LLM judge scores meaning. "
          "Dev numbers are directional.)")


if __name__ == "__main__":
    main()
