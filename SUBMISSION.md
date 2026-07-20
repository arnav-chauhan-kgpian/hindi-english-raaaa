# SUBMISSION — Builderr Speech-to-Text (Streaming / Dictation Track)

## Track
Streaming dictation, scored on the pinned **Apple-silicon MacBook Pro M1** box. The sealed
harness (`solution/stream_server.py`) feeds audio in real time and calls the entry point
[`solution/draft.py`](solution/draft.py): `draft()` + `draft_reset()`.

## Design — one fast+faithful model (no draft/finalizer race)
A single **Whisper-Hindi2Hinglish** model transcribes the rolling audio prefix for the
partials **and** produces the final. Because the same faithful model drives the partials, the
committed text is romanized Hinglish that keeps the Hindi-English code-switch — it never shows
the English-translated draft the reference loses points on.

| Contract concern | How `draft.py` handles it |
| --- | --- |
| `audio_buffer` (cumulative PCM s16le 16k) | `int16 → float32/32768` |
| Partials + final | the **same** Whisper-Hinglish model on the rolling prefix / full buffer |
| **Real-time safety (end-to-final)** | **adaptive duty-cycle throttle**: a new partial decode starts only after 2.5× the *measured* duration of the last one (floor 1 s). Decode time therefore stays well under real time, so the server is never in debt when `end` arrives and the final is a single fresh pass. Without this, re-decoding the whole rolling buffer every ~0.7 s exceeds real time and the final lands late or never. |
| `stable_chars` / churn | commit the **longest common word-prefix of consecutive decodes** (LocalAgreement-2); monotonic — only ever extended. Measured churn **0.000**. |
| Meaning & fidelity | final = the Whisper-Hinglish transcript (romanized Hinglish, code-switch kept, not translated) — the same convention as the RambleFix reference finalizer |
| READY / cold-start | model warms at import (server reaches READY fast); decodes await the load so early partials aren't empty |
| Reliability (no blank/loop/hang) | fully exception-wrapped; the in-stream warm wait is **bounded (45 s)** so a stalled load degrades to committed text instead of hanging the connection (a dropped clip scores 0); final falls back to the last good draft |

## Models (declared + licenses)
| Role | Model | Backend (M1) | License |
| --- | --- | --- | --- |
| Streaming partials + final | `Oriserve/Whisper-Hindi2Hinglish-Apex` (~800M) | transformers, Apple MPS (fp16), CPU fallback | Apache-2.0 |
| Batch entry only (`transcribe.py`) | `faster-whisper small` | CTranslate2, CPU int8 | MIT |

Both are commercial-friendly. The Hinglish model is **standard Whisper architecture**, so it
loads through the ordinary `transformers` ASR pipeline on Apple MPS with no custom code (the
reason it runs on the M1). Swap via `STT_HINGLISH_MODEL`: **Apex** (~800M, default, balanced) ·
**Prime** (large-v3, max fidelity, slower) · **Swift** (72M, fastest).

## Verified on real Apple silicon (GitHub Actions `macos-14`, offline)
The `verify-macos` workflow reproduces the scoring flow (install → warm cache → **block
network** → run) on an arm64 Mac. It confirms:
- `stream_server` imports `draft` + `draft_reset` → **server reaches READY**; wire-contract test passes (`READY port=…`, non-decreasing `stable_chars`).
- Model **loads offline**; **finals are non-blank faithful romanized Hinglish**
  (e.g. `Liber office impress mein ek prastuti document banaana aur buniyaadi formatting ke is spoken tutorial mein aapka svaagat`).
- **Every clip commits useful faithful partials** (committed prefix grows monotonically) — clears the no-useful-partial cap.

> Latency note: the CI VM does not expose Metal for compute (torch falls to CPU there), so the
> CI validates correctness, not speed. The reference (RambleFix) uses the identical
> `device="mps" if available else "cpu"` torch path and runs on the real box, so the pinned M1
> (accelerator on) runs the model on the GPU; Apex is smaller than the large-v3 reference.

## Offline / local-only
- Local-cache-first (`local_files_only=True`); the cached model loads from its on-disk snapshot
  **path** (not the repo id), so the scored run makes zero network calls under `offline_guard`.
- Warm the cache once with network available; the scored run is fully offline.

## Exact commands
```bash
pip install -r requirements.txt -r requirements-streaming.txt   # streaming deps (adds websockets)
# warm the model cache once, network available:
python -c "import solution.transcribe as T; T.get_hinglish_model()"
# offline streaming preview (sealed harness + streaming scorecard):
python preview_stream.py
```

## Notes
- Streaming entry point: `solution/draft.py` — `draft(audio_buffer, is_final) -> (text, stable_chars)` and `draft_reset()`.
- Output is **romanized Hinglish** (Hindi romanized, English terms kept in English) — faithful to
  the code-switch, not translated to English.
- A separate batch entry (`solution/transcribe.py`) also exists for the batch track.
