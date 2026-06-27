# SUBMISSION — Builderr Speech-to-Text (Streaming / Dictation Track)

## Track
Streaming dictation, scored on the frozen **Apple-silicon MacBook Pro M1** box. Audio is fed
in real time via the sealed `stream_server.py` harness; the entry point is
[`solution/draft.py`](solution/draft.py).

## Entry point — contract mapping
`draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]`
(full contract: [`docs/STREAMING_CONTRACT.md`](docs/STREAMING_CONTRACT.md))

| Contract concern | How `draft.py` handles it |
| --- | --- |
| `audio_buffer` (cumulative PCM s16le 16k) | decoded `int16 → float32/32768` |
| Partials / TTFS | faster-whisper-small (CPU int8), debounced ~0.45 s, emitted as `(text, stable_chars)` |
| `stable_chars` / revision churn | longest common-prefix backed off to a word boundary, **monotonic** (never un-committed) |
| End-to-final latency | the Hinglish (Whisper) final + **speculative final**: it is pre-run during the trailing pause so is_final returns near-instantly. Fail-safe: timeout-bounded lock (no hang), synchronous + committed-text fallbacks (no blank), one MPS call at a time, never slower than synchronous. Off via `STT_SPECULATIVE_FINAL=0`. |
| Meaning & fidelity / critical facts | final = **always** Whisper-Hinglish (MPS) → vocab/repair → strip; never gated behind the router, so it can't go blank if another model fails |
| Reliability (no blank/loop/hang) | background model warmup; every path exception-wrapped |

## Models
| Role | Model | Backend (M1) | Precision | License |
| --- | --- | --- | --- | --- |
| Fast ASR / partials | `faster-whisper small` | CTranslate2 (CPU) | int8 | MIT |
| Hinglish ASR / final | `Oriserve/Whisper-Hindi2Hinglish-Apex` (~800M) | transformers (Apple MPS) | fp16 | Apache-2.0 |

The Hinglish model is **standard Whisper architecture** (loads via the ordinary `transformers`
ASR pipeline — the reason it runs on the M1 where the custom qwen3-asr arch did not). Default
**Apex**; override with `STT_HINGLISH_MODEL` (Prime = large-v3, max fidelity, slower).

## Verified (Kaggle T4 GPU — proxy for the M1's MPS; load is device-independent)
- Model **loads + produces faithful romanized Hinglish** (e.g. `…ek prastuti document banaana
  aur buniyaadi formatting ke is spoken tutorial mein aapka svaagat`).
- **Apex** streaming end-to-final: **0.39–1.21 s** across the 6 samples (vs Prime 1.4–5.1 s) —
  Apex matched Prime's fidelity at ~4× the speed, so it's the default to stay under the latency
  caps on the slower M1 GPU.

## Backend / accelerator
- **Apple silicon (scoring box):** Whisper-Hinglish on **MPS** (Metal, fp16), loaded on CPU then
  `.to("mps")`. faster-whisper partials run CPU int8.
- **CUDA box:** the same model loads on `cuda:0` (fp16) — used for the Kaggle verification.
- **Pure CPU:** runs fp32 (slower). `STT_DISABLE_CPU_QWEN=1` skips it → fast-model draft only.

## Reliability / constraints
- Blank-by-crash: none (every path wrapped; final never blank when any candidate has content).
- Hang: none (background warmup, load watchdog, no blocked-network retries).
- Offline after warmup: PASS. Models warm at module import (background thread) so the load
  happens during the network-up setup phase; the cached Qwen is then loaded from its on-disk
  snapshot **path** (not the repo id) so the scored run makes zero HF API calls when the
  network guard is active (`local_files_only=True`).
- Hindi preserved · no romanization · no translation · English tech terms preserved ·
  no Arabic/Urdu leakage · ensemble disabled. Qwen prefers GPU/MPS, CPU fallback allowed.

## Exact commands
```bash
# install (Apple-silicon box)
pip install -r requirements.txt

# warm the model cache once, network available:
HF_HUB_OFFLINE=0 python -m solution.transcribe --debug

# streaming smoke test (feeds a sample wav as 200 ms frames, prints partials + final):
python -m solution.draft samples/<clip>.wav

# batch engine (shared accuracy pipeline), fully offline:
HF_HUB_OFFLINE=1 python preview.py
```

## Notes
- Streaming entry point: `solution/draft.py` (`draft()` + optional `warmup()`).
- Batch engine: `solution/transcribe.py` — shared by `draft.py` for the models and the
  accuracy pipeline (`transcribe(wav, mode)` and `python -m solution.transcribe`).
- Latency depends on the M1 box and is measured by the official harness; the design targets
  a Hinglish final near the Qwen single-pass time (fast pass skipped on escalated clips) and
  TTFS at the first debounced partial. Validate on the frozen box with the harness.
