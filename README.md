# arnav's Speech-to-Text — Local Hinglish Streaming Dictation

A local, offline streaming dictation engine for English, Indian-English, and Hindi-English
(code-switch) work dictation. It preserves the code-switch faithfully — Hindi stays in
Devanagari, English tech terms stay in Latin — and never translates or romanizes.

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

- **Partials (streaming):** faster-whisper-small (CPU int8) on the growing buffer, debounced
  (~0.45 s). Commit a monotonic word-boundary prefix → fast TTFS, low revision churn.
- **Final (`is_final=True`):** the validated accuracy pipeline — recall router → Qwen3-ASR on
  Apple **MPS** only for Hinglish → vocab/repair → Arabic strip. The route is decided *during*
  streaming (sticky), so a Hinglish final goes straight to Qwen and skips a redundant fast pass
  to keep end-to-final latency down.
- **Warmup:** models load in a background thread on the first call (and via `draft.warmup()`),
  so the one-time load never blocks the stream. Every path is exception-wrapped — never
  blank-by-crash, never hang.

## Architecture

```
PCM s16le 16k  (cumulative buffer, 20 ms frames)
 → [streaming] faster-whisper-small int8 (CPU) → monotonic stable-prefix partials
 → [is_final]  recall-biased router
                 → [English]  faster-whisper final
                 → [Hinglish] Qwen3-ASR 0.6B (Apple MPS, transformers fp16, sdpa)
 → normalize_tech_words()   (canonical tech terms; never touches Devanagari)
 → repair_common_asr_errors()
 → (text, stable_chars)
```

The batch engine [`solution/transcribe.py`](solution/transcribe.py) (`transcribe(wav, mode)`
+ CLI) is shared by `draft.py` for models and the accuracy pipeline; on a CUDA box it
auto-uses the vLLM backend, on Apple silicon it uses MPS.

## Models

| Role | Model | Backend (M1) | License |
| --- | --- | --- | --- |
| Fast ASR / partials | `faster-whisper small` (int8) | CTranslate2, CPU | MIT |
| Hinglish ASR / final | `moorlee/qwen3-asr-0.6b-hinglish` (0.6B) | transformers, Apple MPS, fp16 | Apache-2.0 |

No ensemble, no romanization, no translation, no large-v3 fallback, no CPU Qwen,
no AWQ/GPTQ, no CPU quantization. All permanently disabled.

## Accelerator handling

- **Apple silicon (M1/M2/M3):** Qwen runs on the **MPS** (Metal) backend, fp16, `sdpa`
  attention. faster-whisper runs CPU int8 (CTranslate2 has no Metal backend).
- **CUDA box:** the loader auto-selects vLLM (Triton attention) or transformers bf16/FA2.
- **Pure CPU (no accelerator):** the Qwen path is skipped (RULE 7) and the fast English draft
  is returned — never a multi-minute CPU load.

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

- **Scoring box:** Apple-silicon MacBook Pro M1 (no cloud GPU).
- **Requirement:** Apple silicon for the MPS Qwen path; CPU-only still works (fast English
  draft, Hinglish path skipped). On a CUDA box add `pip install vllm` to use the GPU backend.

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
