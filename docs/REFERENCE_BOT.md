# The reference bot — how RambleFix was built

RambleFix is the engine you're trying to beat. It was built by the person running this
challenge, and it's the current **bar to beat** on the board. This page is the *approach* —
not the source. Read it to understand what works, then build your own (better) version.

## What it's trying to do (the objectives)

In order of what matters:

1. **Keep the meaning.** Whatever the person actually meant has to survive — that's the
   real score, not raw word-matching.
2. **Stay faithful to the mix.** When someone speaks Hindi+English, write down *what they
   said* — keep the code-switch. Do **not** quietly translate it all into English. (That's
   the trap free tools fall into, and it's an automatic score cap here.)
3. **Be fast.** Dictation only becomes a habit if it feels instant. Quick English should
   come back in well under a second.
4. **Stay local + offline.** Nothing leaves the machine. That's the whole reason this tool
   needs to exist — companies block cloud dictation.
5. **Never embarrass itself.** No blank outputs, no hangs, no repeating-gibberish loops.
6. **Be shippable.** Small models, runs on a normal computer, licenses that let you release
   it for free.

## How it's built (the shape that works)

It's not one model — it's a small pipeline:

1. **A fast recognizer, always warm.** A resident `whisper.cpp` small model running as a
   local server, so the common English / Indian-English clip comes back sub-second. This
   handles most of the traffic cheaply.

2. **A router.** For each clip it decides: is this plain English (use the fast path) or is it
   Hindi-mixed / risky (escalate)? **This is the crux.** If you run the heavy model on every
   clip, your latency dies; if you never escalate, the mix comes out wrong. Getting the routing
   right is most of the game.

3. **A stronger Hindi-capable path** — a code-switch model (e.g. a Qwen3-ASR / Indic-style
   model) used **only on the escalated clips**. That's how you get the mix right without paying
   the cost on everything.

4. **A finalizer.** Cleans up the result while keeping the code-switch faithful, double-checks
   the things that flip meaning (numbers, dates, "not", names, work terms), and has a
   **repetition guard + a fast fallback** so it never loops or hangs. If everything else fails,
   it falls back to a plain non-translated transcript rather than returning nothing.

Throughout, it keeps the raw candidates + timings + which model ran, so every decision is
auditable (that's also part of the score here).

## Where it actually lands (your bar)

Measured in a live head-to-head against the best free engines, same clips:

| | RambleFix | Best free tool |
|---|---|---|
| **English** | word-error ~**0.04**, ~2s | ~0.06, ~0.45s (faster) |
| **Hindi+English** | word-error ~**0.12**, meaning ~**0.84** | ~0.91 — they translate the mix away |

Read that carefully: on **English it's basically tied** with free tools (and a bit slower).
On the **mix it's ~8× more faithful** — because the free tools translate Hindi into English and
lose what was actually said. **That gap is the whole opportunity.**

## What we learned the hard way (use these)

- **English is close to solved.** Don't pour your weekend into it — grab a good local model and
  move on. The ranking is decided on the mix.
- **The router is where you win or lose.** Run the heavy model only when a clip needs it.
- **Never translate to win meaning.** It looks fine on a meaning score and fails faithfulness —
  and the scorecard caps it. Keep the words people said.
- **Reliability counts as much as accuracy.** A tool that occasionally hangs or loops doesn't get
  used. Guard against it.
- **Offline packaging + licensing is real work.** Pick small, permissive models early — a model
  you can't ship in a free product can't win here.

## How you beat it

You don't need RambleFix's code. You need to push the **mix** past it — meaning **≥0.90**
(it's at ~0.84) while staying **faithful** and **fast**, and matching it on English. That usually
means a better router, a better finalizer, or a better permissive Hindi model than it uses today.
Full targets are in the [README](../README.md).
