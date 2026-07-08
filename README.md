# arnav's Speech-to-Text — Local Hinglish Streaming Dictation

A local, offline streaming dictation engine for English, Indian-English, and Hindi-English
(code-switch) work dictation. The faithful final is produced by a Whisper-large-v3 Hinglish
finetune that keeps the Hindi-English mix as readable romanized Hinglish (e.g. "mujhe kal AWS
ka deployment dekhna hai"), never translating to pure English.

Built for the **streaming / dictation track**: scored on a frozen **Apple-silicon MacBook Pro
M1** (no cloud GPU). Audio is fed in real time; the score is driven by how fast a clean final
lands after you stop talking, plus how faithfully the Hindi-English mix is kept.

## Streaming entry point

[`solution/draft.py`](solution/draft.py) implements the sealed-harness contract
([`docs/STREAMING_CONTRACT.md`](docs/STREAMING_CONTRACT.md)):

```python
def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    # audio_buffer : cumulative PCM s16le, mono, 16 kHz
    # is_final     : False while streaming, True once the user stops
    # returns      : (text_so_far, stable_chars)  — stable_chars = committed prefix length
```

- **One model for partials AND final:** the same Whisper-Hindi2Hinglish model (Apple **MPS**)
  transcribes the rolling prefix for the partials and the full buffer for the final — so the
  committed text is romanized Hinglish that keeps the code-switch, never the English-translated
  draft. No separate draft model, no draft/finalizer race.
- **Commit (`stable_chars`):** the longest common word-prefix of consecutive decodes
  (LocalAgreement-2), monotonic — only ever extended. Low revision churn, faithful partials.
- **READY / cold-start:** the model warms at import (server reaches READY) and every decode
  awaits the load, so the first clip's partials aren't empty while it loads.
- **Reliability:** every path is exception-wrapped; the final falls back to the last good draft
  and never blanks a committed prefix — no crash, hang, or blank.

## Architecture

```
PCM s16le 16k  (cumulative buffer, 20 ms frames)
 → Whisper-Hindi2Hinglish (Apple MPS, transformers) on the rolling prefix   [partials]
 → commit longest common word-prefix of consecutive decodes (monotonic)
 → same model on the full buffer at is_final                                 [final]
 → (text_so_far, stable_chars)      # romanized Hinglish, code-switch kept
```

A separate batch engine [`solution/transcribe.py`](solution/transcribe.py)
(`transcribe(wav, mode)` + CLI) exists for the batch track (it additionally uses
faster-whisper for its fast path).

## Models

| Role | Model | Backend (M1) | License |
| --- | --- | --- | --- |
| Streaming partials + final | `Oriserve/Whisper-Hindi2Hinglish-Apex` (~800M) | transformers, Apple MPS, fp16 | Apache-2.0 |
| Batch entry only (`transcribe.py`) | `faster-whisper small` (int8) | CTranslate2, CPU | MIT |

The Hinglish model is **standard Whisper architecture**, so it loads through the ordinary
`transformers` ASR pipeline on Apple MPS with no custom code — the reason it runs on the M1
where the earlier custom-architecture model (qwen3-asr) did not load. Default is **Apex**
(~800M): on a Kaggle T4 it matched Prime's Hinglish fidelity at ~4× the speed (end-to-final
0.4–1.2 s vs 1.4–5.1 s). Override via `STT_HINGLISH_MODEL`: **Prime** (large-v3, max fidelity,
slower) · **Apex** (balanced, default) · **Swift** (72M, fastest, lower fidelity).

## Accelerator handling

- **Apple silicon (M1/M2/M3):** Whisper-Hinglish runs on the **MPS** (Metal) backend (fp16),
  loaded on CPU then moved with `.to("mps")`; falls back to CPU if MPS is unavailable.
- **CUDA box:** the same model loads on `cuda:0` (fp16) — used for the earlier Kaggle checks.
- **Pure CPU (no accelerator):** runs on CPU (fp32) — slower but functional.

## Offline Mode

- Models load **local-cache-first** (`local_files_only=True`); no cloud during scoring.
- Warm the cache once with network available, then run with `HF_HUB_OFFLINE=1`.
- If the cache is missing during a blocked run, the engine **fails gracefully** (returns the
  fast-path text) and **never hangs** (load watchdog + no blocked-network retries).

## Competition Constraints

- Hindi preserved verbatim (Devanagari never romanized/translated/replaced).
- English tech terms preserved/canonicalized (AWS, GPT, Docker, Kubernetes, PRD, p95, …).
- Numbers and negation preserved.
- Fully local; no outbound network during the scored run.
- Commercial-friendly licenses (MIT + Apache-2.0).

## Hardware

- **Scoring box:** Apple-silicon MacBook Pro M1 — Qwen runs on the integrated GPU (Metal/MPS).
- **Runs anywhere:** Apple silicon uses MPS; a CPU-only box (e.g. a Windows dev PC) runs Qwen
  on CPU (fidelity-first, slower) so you can develop/test it locally; on a CUDA box add
  `pip install vllm` for the GPU backend. `STT_DISABLE_CPU_QWEN=1` skips CPU Qwen for speed.

## Submission Notes

```bash
pip install -r requirements.txt           # Apple-silicon box (CPU faster-whisper + MPS torch)
# warm the model cache once (network available):
HF_HUB_OFFLINE=0 python -m solution.transcribe --debug
# streaming smoke test (no harness needed — feeds a sample wav as 200 ms frames):
python -m solution.draft samples/<clip>.wav
```

See [`SUBMISSION.md`](SUBMISSION.md) for models, the streaming contract mapping, and the
streaming/latency notes.
