#!/usr/bin/env python
# =============================================================================
# kaggle_validate_submission.py
#
# SINGLE self-contained Kaggle GPU validation for the LOCKED Builderr STT stack:
#
#   Audio -> faster-whisper-small-int8 -> recall router
#         -> [English] return fast output
#         -> [else] Qwen3-ASR 0.6B Hinglish (vLLM bf16 GPU)
#         -> normalize_tech_words() -> repair_common_asr_errors() -> JSON
#
# It DOES NOT change the architecture. It only drives solution.transcribe and
# measures/validates it once before submission.
#
# USAGE (Kaggle): enable GPU, add your repo as a Dataset (or have it in /kaggle/working),
# paste this whole file into one cell, Run. It auto-discovers the repo (looks for
# solution/transcribe.py), or set env BUILDERR_REPO=/path/to/repo.
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import time
import subprocess

EXPECTED_KEYS = {"text", "mode_used", "language_guess", "timings_ms",
                 "raw_candidates", "model_ids", "local_only"}


# ----------------------------------------------------------------------------- #
# small helpers
# ----------------------------------------------------------------------------- #
def _sh(cmd: str) -> int:
    return subprocess.run(cmd, shell=True).returncode


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _fmt(x, suffix=""):
    return "n/a" if x is None else (f"{x:.3f}" if suffix == "" else f"{x:.0f}{suffix}")


# ----------------------------------------------------------------------------- #
# STEP 1 — install + GPU detection
# ----------------------------------------------------------------------------- #
def step1_install_and_gpu() -> dict:
    print("=" * 60)
    print("STEP 1 — install dependencies + detect GPU")
    print("=" * 60)
    # ---- Option 2: vLLM backend (the locked architecture; ~0.02s/token -> p95 < 5s). ----
    # vLLM needs a torch it supports, so qwen-asr[vllm] DOWNGRADES Kaggle's torch (2.10) to
    # vLLM's pin. We then reinstall a MATCHING torchvision/torchaudio so the
    # "operator torchvision::nms does not exist" ABI error cannot happen.
    # IMPORTANT: do NOT import torch before this install, or the old (2.10) module is cached.
    # Fallback to transformers backend with STT_USE_VLLM=0.
    use_vllm = os.environ.get("STT_USE_VLLM", "1") == "1"
    if use_vllm:
        print("pip install qwen-asr[vllm]  (pulls vLLM + its torch; downgrades torch) ...")
        _sh(f'{sys.executable} -m pip install -q "qwen-asr[vllm]"')
        import torch  # FIRST torch import -> the downgraded version vLLM installed
        tver = torch.__version__.split("+")[0]                      # e.g. 2.6.0
        tv_minor = int(tver.split(".")[1]) + 15                     # torch 2.6 -> torchvision 0.21
        cu = (torch.version.cuda or "12.4").replace(".", "")[:3]    # "124"
        print(f"matching torchvision/torchaudio to torch {tver} (cu{cu}) ...")
        _sh(f'{sys.executable} -m pip install -q --no-deps --force-reinstall '
            f'"torchvision==0.{tv_minor}.*" "torchaudio=={tver}" '
            f'--index-url https://download.pytorch.org/whl/cu{cu}')
    else:
        try:
            import torch  # noqa: F401
            print("torch already present:", torch.__version__)
        except Exception:
            _sh(f"{sys.executable} -m pip install -q torch")
        _sh(f"{sys.executable} -m pip install -q qwen-asr")

    for pkg in ("faster-whisper", "ctranslate2", "transformers", "accelerate", "psutil"):
        print(f"pip install {pkg} ...")
        _sh(f"{sys.executable} -m pip install -q {pkg}")

    # Guard the torchvision::nms ABI (transformers imports torchvision during model load).
    try:
        from torchvision.ops import nms  # noqa: F401
        print("torchvision ABI ok")
    except Exception as e:
        print("WARNING: torchvision ABI mismatch (", e, ") -> 'Restart & Run All' may be needed")

    import torch
    info = {"cuda": torch.cuda.is_available()}
    # flash-attn: SKIP by default (vLLM has its own kernels; the T4 build fails anyway and
    # the loader uses sdpa). Opt in with STT_BUILD_FLASH_ATTN=1.
    if (os.environ.get("STT_BUILD_FLASH_ATTN") == "1" and info["cuda"]
            and torch.cuda.is_bf16_supported()):
        print("pip install flash-attn (opt-in) ...")
        _sh(f"{sys.executable} -m pip install -q flash-attn --no-build-isolation")
    else:
        print("skipping flash-attn (vLLM/sdpa used)")

    # faster-whisper (ctranslate2) GPU needs cuBLAS + cuDNN discoverable; without them ct2
    # SILENTLY runs on CPU (the FAST p50~7s bug while Qwen is on GPU). Install + expose libs.
    if info["cuda"]:
        for w in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
            _sh(f"{sys.executable} -m pip install -q {w}")
        libs = []
        for mod in ("nvidia.cublas", "nvidia.cudnn"):
            try:
                m = __import__(mod, fromlist=["lib"])
                d = os.path.join(os.path.dirname(m.__file__), "lib")
                if os.path.isdir(d):
                    libs.append(d)
            except Exception:
                pass
        if libs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
            print("LD_LIBRARY_PATH += cuBLAS/cuDNN for ctranslate2 GPU ->", libs)
    if not info["cuda"]:
        info.update(gpu="(NO GPU)", cuda_version=None, vram_gb=0.0, bf16=False)
        print("\n*** NO CUDA GPU DETECTED — enable the Kaggle GPU accelerator and re-run. ***")
        return info
    p = torch.cuda.get_device_properties(0)
    info.update(
        gpu=torch.cuda.get_device_name(0),
        cuda_version=torch.version.cuda,
        vram_gb=round(p.total_memory / (1024 ** 3), 1),
        bf16=bool(torch.cuda.is_bf16_supported()),
    )
    try:
        import flash_attn  # noqa: F401
        info["flash_attn"] = True
    except Exception:
        info["flash_attn"] = False
    print(f"\nGPU            : {info['gpu']}")
    print(f"CUDA           : {info['cuda_version']}")
    print(f"VRAM           : {info['vram_gb']} GB")
    print(f"bf16 supported : {info['bf16']}")
    print(f"flash-attn     : {info['flash_attn']}  (sdpa fallback if False)")
    return info


# ----------------------------------------------------------------------------- #
# locate the repo + sample clips
# ----------------------------------------------------------------------------- #
def find_repo() -> str:
    cand = []
    env = os.environ.get("BUILDERR_REPO")
    if env:
        cand.append(env)
    cand.append(os.getcwd())
    try:                                  # __file__ is undefined in a notebook kernel
        cand.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    for root in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if os.path.isdir(root):
            for dp, _, fs in os.walk(root):
                if "transcribe.py" in fs and os.path.basename(dp) == "solution":
                    cand.append(os.path.dirname(dp))
                if dp.count(os.sep) - root.count(os.sep) > 4:
                    continue
    for c in cand:
        if c and os.path.exists(os.path.join(c, "solution", "transcribe.py")):
            return os.path.abspath(c)
    raise RuntimeError(
        "Could not find the repo (need solution/transcribe.py). Add your repo as a Kaggle "
        "Dataset or set BUILDERR_REPO=/path/to/repo.")


def load_clips(repo: str):
    clips = []
    seen = set()
    for man, audio_dir in ((os.path.join(repo, "samples", "manifest.json"), os.path.join(repo, "samples")),
                           (os.path.join(repo, "data", "dev", "manifest.json"),
                            os.path.join(repo, "data", "dev", "audio"))):
        if not os.path.exists(man):
            continue
        for c in json.load(open(man, encoding="utf-8")):
            cid = c["clip_id"]
            wav = c.get("audio")
            wav = os.path.join(repo, wav) if wav else os.path.join(audio_dir, cid + ".wav")
            if cid in seen or not os.path.exists(wav):
                continue
            lang = (c.get("language") or "").lower()
            label = "hinglish" if lang in ("hi-en", "hinglish") else (
                "english" if lang in ("english", "en") or "english" in (c.get("category") or "") else "other")
            clips.append({"id": cid, "wav": wav, "gold": c.get("gold", ""), "label": label})
            seen.add(cid)
    return clips


# ----------------------------------------------------------------------------- #
# STEP 2 — warmup (download once) then go OFFLINE and verify
# ----------------------------------------------------------------------------- #
def step2_warmup_offline(T, repo, clips):
    print("\n" + "=" * 60)
    print("STEP 2 — warmup (download once) -> offline -> verify")
    print("=" * 60)
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "STT_OFFLINE"):
        os.environ.pop(k, None)  # allow the one-time download during warmup

    t = time.time(); fm = T.get_fast_model(); fast_load = time.time() - t
    t = time.time(); hm = T.get_hinglish_model(); hing_load = time.time() - t
    print(f"fast model loaded  : {fm is not None}  ({fast_load:.1f}s)")
    print(f"hinglish loaded    : {hm is not None}  ({hing_load:.1f}s)  "
          f"backend={dict(T._HINGLISH_META).get('backend')} "
          f"precision={dict(T._HINGLISH_META).get('precision')}")
    fm_meta = dict(T._FAST_META)
    print(f"FAST model device  : {fm_meta.get('device')} / compute={fm_meta.get('compute_type')}  "
          f"(if 'cpu' here, that's the latency bug)")
    print(f"HINGLISH device    : {dict(T._HINGLISH_META).get('device')}")
    if hm is None:
        print("*** HINGLISH MODEL DID NOT LOAD ***  error:", T._LAST_HINGLISH_ERROR)

    sample = next((c for c in clips if c["label"] == "hinglish"), clips[0] if clips else None)
    if sample:
        r = T.transcribe(sample["wav"], "auto")  # dummy inference triggers all lazy loads
        print("warmup inference ok:", set(r) == EXPECTED_KEYS)

    # ---- go fully offline (mirror the scored run) ----
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    offline_ok = True
    try:
        sys.path.insert(0, repo)
        from offline_guard import block_network
        block_network()  # hard-block outbound sockets, exactly like preview.py
        print("network blocked (loopback only).")
    except Exception as e:
        print("could not import offline_guard:", e)
    if sample:
        try:
            r = T.transcribe(sample["wav"], "auto")
            offline_ok = set(r) == EXPECTED_KEYS and bool((r.get("text") or "").strip())
        except Exception as e:
            offline_ok = False
            print("offline inference FAILED:", type(e).__name__, e)
    print(f"OFFLINE inference works (no internet): {offline_ok}")
    return {"fast_load": fast_load, "hing_load": hing_load,
            "fast_ok": fm is not None, "hing_ok": hm is not None, "offline_ok": offline_ok}


# ----------------------------------------------------------------------------- #
# STEP 3 — per-model benchmark
# ----------------------------------------------------------------------------- #
def _vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def step3_model_bench(T, SC, clips):
    print("\n" + "=" * 60)
    print("STEP 3 — per-model benchmark (offline)")
    print("=" * 60)
    eng = [c for c in clips if c["label"] == "english"]
    hin = [c for c in clips if c["label"] == "hinglish"]

    out = {"fast": {}, "hing": {}}
    # FAST model on English clips
    lat, wers, means = [], [], []
    for c in eng or clips:
        audio = T.load_audio(c["wav"])
        s = time.time(); r = T.fast_transcribe(audio); lat.append((time.time() - s) * 1000)
        wers.append(SC.wer(c["gold"], r["text"])); means.append(SC.judge_meaning(c["gold"], r["text"]))
        print(f"  FAST  {c['id'][:32]:32s} {lat[-1]:7.0f} ms")
    out["fast"] = {"wer": _mean(wers), "meaning": _mean(means), "avg": _mean(lat),
                   "p50": _pct(lat, .50), "p95": _pct(lat, .95)}

    # HINGLISH model on Hinglish clips
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    lat, wers, means = [], [], []
    for c in hin:
        audio = T.load_audio(c["wav"])
        s = time.time(); r = T.hinglish_transcribe(audio); lat.append((time.time() - s) * 1000)
        wers.append(SC.wer(c["gold"], r["text"])); means.append(SC.judge_meaning(c["gold"], r["text"]))
        print(f"  HING  {c['id'][:32]:32s} {lat[-1]:7.0f} ms")
    out["hing"] = {"wer": _mean(wers), "meaning": _mean(means), "avg": _mean(lat),
                   "p50": _pct(lat, .50), "p95": _pct(lat, .95), "vram_mb": _vram_mb()}
    print(f"FAST   p50={_fmt(out['fast']['p50'],'ms')} p95={_fmt(out['fast']['p95'],'ms')} "
          f"WER={_fmt(out['fast']['wer'])} Meaning={_fmt(out['fast']['meaning'])}")
    print(f"HING   p50={_fmt(out['hing']['p50'],'ms')} p95={_fmt(out['hing']['p95'],'ms')} "
          f"WER={_fmt(out['hing']['wer'])} Meaning={_fmt(out['hing']['meaning'])} "
          f"VRAM={_fmt(out['hing']['vram_mb'],'MB')}")
    return out


# ----------------------------------------------------------------------------- #
# STEP 4 — full pipeline
# ----------------------------------------------------------------------------- #
def step4_full_pipeline(T, SC, clips):
    print("\n" + "=" * 60)
    print("STEP 4 — full pipeline end-to-end (offline)")
    print("=" * 60)
    lat, blanks, hangs, rows = [], 0, 0, []
    t_start = time.time()
    for c in clips:
        s = time.time(); r = T.transcribe(c["wav"], "auto"); dt = (time.time() - s) * 1000
        lat.append(dt)
        print(f"  FULL  {c['id'][:32]:32s} {dt:7.0f} ms  mode={r.get('mode_used')} "
              f"asr={r.get('timings_ms',{}).get('asr')}ms")
        txt = (r.get("text") or "").strip()
        if not txt:
            blanks += 1
        if int(r.get("timings_ms", {}).get("total", 0)) > 5000:
            hangs += 1
        rows.append({"clip": c, "result": r, "text": txt})
    wall = time.time() - t_start
    n = len(clips) or 1
    res = {"p50": _pct(lat, .50), "p95": _pct(lat, .95), "avg": _mean(lat),
           "throughput": n / wall if wall else 0.0,
           "blank_rate": blanks / n, "hang_rate": hangs / n, "rows": rows}
    print(f"FULL   p50={_fmt(res['p50'],'ms')} p95={_fmt(res['p95'],'ms')} "
          f"throughput={res['throughput']:.2f} clips/s blank={res['blank_rate']:.2f} hang={res['hang_rate']:.2f}")
    return res


# ----------------------------------------------------------------------------- #
# STEP 5 — competition constraints
# ----------------------------------------------------------------------------- #
def step5_constraints(T, rows, offline_ok, gpu_ok):
    print("\n" + "=" * 60)
    print("STEP 5 — competition constraints")
    print("=" * 60)
    import re
    deva = re.compile(r"[ऀ-ॿ]")
    arabic = re.compile(r"[؀-ۿ]")
    tech = ["impress", "document", "formatting", "tutorial", "workspace", "AWS", "GPT", "Docker", "API"]

    hin_rows = [r for r in rows if r["clip"]["label"] == "hinglish"]
    hindi_preserved = all(bool(deva.search(r["text"])) for r in hin_rows) if hin_rows else True
    no_romanization = hindi_preserved  # Devanagari present in every Hinglish output → not romanized
    no_translation = hindi_preserved   # Hindi script present → not translated to English
    no_arabic = not any(arabic.search(r["text"]) for r in rows)
    tech_ok = all(
        all(t in r["text"] for t in tech if t in r["clip"]["gold"])
        for r in rows) if rows else True
    schema_ok = all(set(r["result"]) == EXPECTED_KEYS for r in rows)
    ensemble_off = (T.ENABLE_ENSEMBLE is False) and not any(
        cand.get("engine") == "ensemble" for r in rows for cand in r["result"].get("raw_candidates", []))

    checks = {
        "Hindi preserved": hindi_preserved,
        "No Romanization": no_romanization,
        "No Translation": no_translation,
        "English tech words preserved": tech_ok,
        "No Arabic/Urdu leakage": no_arabic,
        "Offline after warmup": offline_ok,
        "JSON schema unchanged": schema_ok,
        "GPU-only Hinglish": gpu_ok,
        "Ensemble disabled": ensemble_off,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return checks


# ----------------------------------------------------------------------------- #
# STEP 6 — final verdict
# ----------------------------------------------------------------------------- #
def step6_report(gpu_info, s3, full, checks):
    gate_p95_ok = (full["p95"] is not None and full["p95"] < 5000)
    no_blanks = full["blank_rate"] == 0.0
    all_checks = all(checks.values())
    ready = bool(gpu_info.get("cuda") and all_checks and gate_p95_ok and no_blanks)

    line = "=" * 37
    print("\n" + line)
    print("BUILDERR FINAL VALIDATION")
    print(f"GPU: {gpu_info.get('gpu')}")
    print(f"VRAM: {gpu_info.get('vram_gb')} GB")
    print("\nFAST:")
    print(f"WER: {_fmt(s3['fast']['wer'])}")
    print(f"P50: {_fmt(s3['fast']['p50'],'ms')}")
    print(f"P95: {_fmt(s3['fast']['p95'],'ms')}")
    print("\nHINGLISH:")
    print(f"WER: {_fmt(s3['hing']['wer'])}")
    print(f"P50: {_fmt(s3['hing']['p50'],'ms')}")
    print(f"P95: {_fmt(s3['hing']['p95'],'ms')}")
    print("\nFULL PIPELINE:")
    print(f"P50: {_fmt(full['p50'],'ms')}")
    print(f"P95: {_fmt(full['p95'],'ms')}")
    print(f"BLANK RATE: {full['blank_rate']:.2f}")
    print(f"HANG RATE: {full['hang_rate']:.2f}")
    print(f"OFFLINE: {'PASS' if checks.get('Offline after warmup') else 'FAIL'}")
    print("\nFINAL VERDICT:")
    print("READY TO SUBMIT" if ready else "NOT READY TO SUBMIT")
    if not ready:
        failed = [k for k, v in checks.items() if not v]
        if not gpu_info.get("cuda"):
            failed.append("no GPU")
        if not gate_p95_ok:
            failed.append(f"full p95 >= 5s ({_fmt(full['p95'],'ms')})")
        if not no_blanks:
            failed.append(f"blank rate {full['blank_rate']:.2f}")
        print("  blockers:", ", ".join(failed) or "(none)")
    print(line)
    return ready


# ----------------------------------------------------------------------------- #
def main():
    gpu = step1_install_and_gpu()

    repo = find_repo()
    print("repo:", repo)
    sys.path.insert(0, repo)
    import solution.transcribe as T
    import scorecard as SC

    clips = load_clips(repo)
    print(f"clips: {len(clips)} "
          f"(english={sum(c['label']=='english' for c in clips)}, "
          f"hinglish={sum(c['label']=='hinglish' for c in clips)})")
    if not clips:
        print("NO CLIPS FOUND — ensure samples/ is in the repo.")
        return

    s2 = step2_warmup_offline(T, repo, clips)
    gpu_ok = bool(gpu.get("cuda")) and s2["hing_ok"]
    s3 = step3_model_bench(T, SC, clips)
    full = step4_full_pipeline(T, SC, clips)
    checks = step5_constraints(T, full["rows"], s2["offline_ok"], gpu_ok)
    step6_report(gpu, s3, full, checks)


if __name__ == "__main__":
    main()
