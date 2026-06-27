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

- **Partials (streaming):** faster-whisper-small (CPU int8) on the growing buffer, debounced
  (~0.45 s). Commit a monotonic word-boundary prefix → fast TTFS, low revision churn.
- **Final (`is_final=True`):** **always** the Whisper-Hinglish model (Apple **MPS**) →
  vocab/repair → strip. It is *not* gated behind the router, so a non-blank faithful final
  never depends on any other model loading. Fast-model text is only a degradation fallback.
- **Speculative final (latency):** when the speaker pauses near the end, the Whisper-Hinglish
  pass is launched in the background *before* `is_final` arrives, so the final returns
  near-instantly. Fail-safe by construction — never hangs (timeout-bounded lock), never
  blanks/crashes (synchronous + committed-text fallbacks), never runs two MPS calls at once,
  never slower than the plain synchronous path. Disable with `STT_SPECULATIVE_FINAL=0`.
- **Warmup:** models load in a background thread on import / first call (and via
  `draft.warmup()`), so the one-time load never blocks the stream. Every path is
  exception-wrapped — never blank-by-crash, never hang.

## Architecture

```
PCM s16le 16k  (cumulative buffer, 20 ms frames)
 → [streaming] faster-whisper-small int8 (CPU) → monotonic stable-prefix partials
 → [is_final]  Whisper-Hindi2Hinglish (large-v3 finetune, Apple MPS, transformers) — always
 → normalize_tech_words()   (canonical tech terms: AWS, GPT, Docker, …)
 → repair_common_asr_errors()
 → (text, stable_chars)
```

The batch engine [`solution/transcribe.py`](solution/transcribe.py) (`transcribe(wav, mode)`
+ CLI) is shared by `draft.py` for models and the accuracy pipeline.

## Models

| Role | Model | Backend (M1) | License |
| --- | --- | --- | --- |
| Fast ASR / partials | `faster-whisper small` (int8) | CTranslate2, CPU | MIT |
| Hinglish ASR / final | `Oriserve/Whisper-Hindi2Hinglish-Prime` (large-v3) | transformers, Apple MPS, fp16 | Apache-2.0 |

The Hinglish model is **standard Whisper architecture**, so it loads through the ordinary
`transformers` ASR pipeline on Apple MPS with no custom code — the reason it runs on the M1
where the earlier custom-architecture model (qwen3-asr) did not load. Pick the variant via
`STT_HINGLISH_MODEL`: **Prime** (2B, most faithful) · **Swift** / **Apex** (smaller, faster).

## Accelerator handling

- **Apple silicon (M1/M2/M3):** Whisper-Hinglish runs on the **MPS** (Metal) backend (fp16),
  loaded on CPU then moved with `.to("mps")`. faster-whisper partials run CPU int8.
- **CUDA box:** the same model loads on `cuda:0` (fp16). 
- **Pure CPU (no accelerator):** runs on CPU (fp32) — slower but functional. Set
  `STT_DISABLE_CPU_QWEN=1` to skip it and return the fast-model draft instead.

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
