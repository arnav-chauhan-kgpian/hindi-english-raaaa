# arnav's Speech-to-Text — Local Hinglish Dictation Engine

A local, offline speech-to-text engine for English, Indian-English, and Hindi-English
(code-switch) work dictation. It preserves the code-switch faithfully — Hindi stays in
Devanagari, English tech terms stay in Latin — and never translates or romanizes.

## Architecture

```
audio
 → faster-whisper-small-int8 (fast English / Indian-English ASR, GPU)
 → recall-biased router
     → [English]  return fast output
     → [else]     Qwen3-ASR 0.6B Hinglish (GPU, vLLM, Triton attention, fp16)
 → normalize_tech_words()   (canonical tech terms; never touches Devanagari)
 → repair_common_asr_errors()
 → JSON
```

The single engine entry point is [`solution/transcribe.py`](solution/transcribe.py):
`transcribe(wav_path, mode) -> dict`, plus the CLI
`python -m solution.transcribe --input clip.wav --mode auto --output result.json`.

## Models

| Role | Model | Backend | License |
| --- | --- | --- | --- |
| Fast ASR | `faster-whisper small` (int8_float16) | CTranslate2, GPU | MIT |
| Hinglish ASR | `moorlee/qwen3-asr-0.6b-hinglish` (0.6B) | vLLM (Triton attn), fp16, GPU | Apache-2.0 |

No ensemble, no romanization, no translation, no large-v3 fallback, no CPU Qwen,
no AWQ/GPTQ, no CPU quantization. All permanently disabled.

## Offline Mode

- Models load **local-cache-first** (`local_files_only=True`); no cloud during scoring.
- Warm the cache once with network available, then run with `HF_HUB_OFFLINE=1`.
- If the cache is missing during a blocked run, the engine **fails gracefully** (returns
  fast-path text or a valid blank contract) and **never hangs** (load watchdog + no
  blocked-network retries).
- The Hinglish model is **GPU-only**: without CUDA it is skipped and the fast-path output
  is returned (never a multi-minute CPU load).

## Latency

Measured on a Tesla T4 (offline, warm):

| Stage | p50 | p95 |
| --- | --- | --- |
| Fast ASR | 256 ms | 323 ms |
| Hinglish ASR (Qwen, vLLM) | 539 ms | 632 ms |
| **Full pipeline** | **905 ms** | **4856 ms** |

Blank rate 0% · hang rate 0%.

## Competition Constraints

- Hindi preserved verbatim (Devanagari never romanized/translated/replaced).
- English tech terms preserved/canonicalized (AWS, GPT, Docker, Kubernetes, PRD, p95, …).
- Numbers and negation preserved.
- Fully local; no outbound network during the scored run.
- Output JSON schema unchanged (`text`, `mode_used`, `language_guess`, `timings_ms`,
  `raw_candidates`, `model_ids`, `local_only`).
- Commercial-friendly licenses (MIT + Apache-2.0).

## Hardware Used

- **Validation:** Kaggle Tesla T4 (16 GB), CUDA 12.x, Linux.
- **Requirement:** Linux + NVIDIA GPU (the Hinglish path runs on GPU via vLLM). VRAM ≈ 2 GB
  model + KV cache; fits any modern GPU. The fast path runs on the GPU as well.

## Submission Notes

```bash
pip install -r requirements.txt           # GPU box; pulls qwen-asr[vllm] + faster-whisper
# warm the model cache once (network available):
HF_HUB_OFFLINE=0 python -m solution.transcribe --debug
# then score fully offline:
HF_HUB_OFFLINE=1 python preview.py
```

See [`SUBMISSION.md`](SUBMISSION.md) for the exact validated commit, models, latencies, and
reproduction commands. vLLM auto-selects the Triton attention backend (set in the loader)
so it runs on pre-Ampere GPUs such as the T4.
