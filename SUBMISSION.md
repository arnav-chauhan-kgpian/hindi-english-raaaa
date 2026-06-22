# SUBMISSION — Builderr Speech-to-Text

## Validated commit
- **Validated commit (READY run):** `2d96a23` (repo `arnav-chauhan-kgpian/hindi-english-raaaa`, branch `main`)
- **Release commit:** this commit — a no-behavior-change finalization (dead-code removal,
  requirements/README/SUBMISSION). The validated latencies and constraints below are
  unchanged by the cleanup.

## Models
| Role | Model | Backend | Precision | License |
| --- | --- | --- | --- | --- |
| Fast ASR | `faster-whisper small` | CTranslate2 (GPU) | int8_float16 | MIT |
| Hinglish ASR | `moorlee/qwen3-asr-0.6b-hinglish` | vLLM (Triton attn) | fp16 | Apache-2.0 |

## Backend
- Hinglish: **vLLM**, `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (set by the loader; works on
  pre-Ampere GPUs like the T4). Falls back to transformers bf16/fp16 if vLLM is unavailable.
- Generation: greedy (`num_beams=1`, `do_sample=False`), `max_new_tokens=256`,
  `no_repeat_ngram_size=3`, `repetition_penalty=1.3` (anti-runaway).

## GPU
- Validated on **Kaggle Tesla T4 (16 GB), CUDA 12.x, Linux**.

## Latencies (offline, warm, Tesla T4)
| Stage | p50 | p95 |
| --- | --- | --- |
| Fast ASR | 256 ms | 323 ms |
| Hinglish ASR | 539 ms | 632 ms |
| **Full pipeline** | **905 ms** | **4856 ms** |

## Reliability
- Blank rate: **0%**
- Hang rate: **0%**
- Offline (network blocked after warmup): **PASS**

## Competition constraints (9/9 PASS)
Hindi preserved · no romanization · no translation · English tech terms preserved ·
no Arabic/Urdu leakage · offline after warmup · JSON schema unchanged · GPU-only Hinglish ·
ensemble disabled.

## Exact commands
```bash
# install (GPU box)
pip install -r requirements.txt

# warm the model cache once, network available:
HF_HUB_OFFLINE=0 python -m solution.transcribe --debug

# score on the dev manifest (online dev preview):
python preview.py

# score fully offline (no internet after warmup — mirrors official scoring):
HF_HUB_OFFLINE=1 python preview.py
```

## Notes
- Engine entry point: `solution/transcribe.py`
  (`transcribe(wav, mode)` and `python -m solution.transcribe --input … --output …`).
- Output JSON keys (unchanged): `text`, `mode_used`, `language_guess`, `timings_ms`,
  `raw_candidates`, `model_ids`, `local_only`.
- The Kaggle validation notebook used to produce the numbers above is
  `kaggle_validate_submission.ipynb`.
