# Streaming Dictation Track — Contract Summary

## Core Function Signature

You implement a single function in `solution/draft.py`:

```python
def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
```

**Parameters:**
- `audio_buffer`: Complete PCM s16le mono audio at 16 kHz sampled so far
- `is_final`: Boolean flag—`False` during streaming, `True` after user stops

**Returns:**
- Tuple of (transcript text, stable character count)
- `text_so_far`: Your best transcription; preserve Hindi-English code-switching
- `stable_chars`: Length of committed prefix you promise never to rewrite

## Audio Specifications

The sealed harness (`stream_server.py`) feeds audio as:
- **Format:** PCM s16le (signed 16-bit little-endian)
- **Sample rate:** 16 kHz
- **Channels:** Mono (1)
- **Chunk cadence:** 20 milliseconds (~640 bytes) per frame, arriving in real time

## Wire Protocol (Reference)

**Evaluator → solution:**
- `{"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,"clip_id":"<opaque>"}`
- Raw binary WebSocket frames (audio chunks, 20ms intervals)
- `{"type":"end"}`

**Solution → evaluator:**
- Zero or more: `{"type":"partial","text":"...","stable_chars":N}`
- Exactly one final: `{"type":"final","text":"..."}`
- Optional audit: `{"type":"meta","model_ids":[...],"local_only":true}`

## Scoring (100 points total)

| Metric | Weight | Threshold |
|--------|--------|-----------|
| Meaning & fidelity | 40 | Judged on final from median-latency run |
| Critical facts/terms | 20 | Numbers, negations, names preserved |
| End-to-final latency | 25 | ≤2000ms → 20+ pts; ≤1000ms → 25 pts |
| Time-to-first-stable (TTFS) | 5 | ≤1000ms → 5 pts |
| Revision churn | 5 | Penalizes rewritten committed tokens |
| Streaming reliability | 5 | No blanks, loops, drops, or hangs |

## Latency Scoring Curve

- **≤1000ms** → 25 points (stretch goal)
- **1000–2000ms** → linear 25→20 (realistic target)
- **2000–3500ms** → linear 20→10 (RambleFix ~4.3s median)
- **>5000ms** → 0 points

**TTFS:** ≤1000ms yields 5 points; >2500ms yields 0.

## Hard Caps

Clip score capped at:
- **70** if no partial arrives before end, or no useful committed partial achieved
- **80** if median latency exceeds 4000ms
- **50** if critical facts flip, or connection drops
- **20** if final is blank or WER >0.9

## Benchmark (RambleFix)

RambleFix—ineligible for the prize—measured on the frozen MacBook Pro M1 box:
- **End-to-final:** p50 4.3s, p95 7.4s (well above 2s target)
- **TTFS:** p50 2.0s (first useful partial still too late)
- **Hindi-English fidelity:** WER 0.21, meaning 0.76

The contract emphasizes: this benchmark is "beatable" if you achieve a final under ~2s while
maintaining meaning >0.90 and first useful partial <~1s.
