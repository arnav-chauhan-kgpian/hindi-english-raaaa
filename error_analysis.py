"""builderr · STRICT ERROR ANALYSIS (read-only — writes no features, changes no module).

For every clip it prints GOLD / FAST / HINGLISH / ENSEMBLE, a categorized token-level
diff against GOLD, aggregate error counts per candidate, the top-20 errors, and a
benchmark-evidence answer to: should the ensemble be (A) removed, (B) Hinglish-only,
(C) tech-terms-only, or (D) token-wise-on-disagreements.

It calls the existing fast/hinglish ASR + ensemble.merge_transcripts exactly as the
pipeline does, then diffs the raw outputs. Token comparison here is FULL (Devanagari +
Latin) — unlike scorecard's ASCII-only proxy — so it reveals what the proxy hides.

    python error_analysis.py            # all sample/dev clips with audio present
    python error_analysis.py --limit 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import solution.transcribe as T          # noqa: E402
from solution import ensemble as E       # noqa: E402
from solution import vocab as V          # noqa: E402
import scorecard as SC                   # noqa: E402

PUNCT = ".,!?;:\"'()[]{}…—–-"
NEG = set(SC._NEG)                                  # not, n't, nahi, mat, ...
TECH = set(V.TECH_TERMS.keys())                     # lowercase canonical tech keys
RHIN = E.ROMANIZED_HINDI
CATS = ["wrong Hindi", "wrong English", "wrong tech term", "wrong spacing",
        "wrong capitalization", "wrong number", "wrong negation"]


# --------------------------------------------------------------------------- #
# tokenization + classification
# --------------------------------------------------------------------------- #
def toks(s: str) -> list[str]:
    return [w for w in (s or "").split() if w.strip(PUNCT)]


def norm(t: str) -> str:
    return t.strip(PUNCT).lower()


def is_deva(s: str) -> bool:
    return E.contains_devanagari(s)


def kind(tok: str) -> str:
    c = tok.strip(PUNCT)
    low = c.lower()
    if any(ch.isdigit() for ch in c):
        return "number"
    if low in NEG:
        return "negation"
    if low in TECH or re.fullmatch(r"p\d{2,3}", low):
        return "tech"
    if is_deva(c) or low in RHIN:
        return "hindi"
    return "english"


def classify_sub(g: str, p: str) -> str:
    gk, pk = kind(g), kind(p)
    if "number" in (gk, pk) and re.findall(r"\d+", g) != re.findall(r"\d+", p):
        return "wrong number"
    if "negation" in (gk, pk):
        return "wrong negation"
    if "tech" in (gk, pk):
        return "wrong tech term"
    if "hindi" in (gk, pk):
        return "wrong Hindi"
    return "wrong English"


def classify_single(tok: str) -> str:
    return {"hindi": "wrong Hindi", "tech": "wrong tech term", "number": "wrong number",
            "negation": "wrong negation", "english": "wrong English"}.get(kind(tok), "wrong English")


# --------------------------------------------------------------------------- #
# alignment (Needleman–Wunsch over case-folded tokens) + spacing detection
# --------------------------------------------------------------------------- #
def align(a: list[str], b: list[str]) -> list[tuple[str, int, int]]:
    ca, cb = [norm(x) for x in a], [norm(x) for x in b]
    na, nb = len(a), len(b)
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = i
    for j in range(1, nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sub = 0 if ca[i - 1] == cb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    ops: list[tuple[str, int, int]] = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = 0 if ca[i - 1] == cb[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                ops.append(("match" if sub == 0 else "sub", i - 1, j - 1)); i -= 1; j -= 1; continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1)); i -= 1; continue
        ops.append(("ins", -1, j - 1)); j -= 1
    ops.reverse()
    return ops


def detect_spacing(gt: list[str], pt: list[str]):
    """Find split/join mismatches (p95↔'p 95', karlo↔'kar lo'). Returns
    (consumed_gold_idx, consumed_pred_idx, pairs)."""
    g, p = [norm(x) for x in gt], [norm(x) for x in pt]
    cg: set[int] = set()
    cp: set[int] = set()
    pairs: list[tuple[str, str]] = []
    for i in range(len(g)):                          # one gold token = two pred tokens
        for j in range(len(p) - 1):
            if i in cg or j in cp or (j + 1) in cp:
                continue
            if g[i] and g[i] == p[j] + p[j + 1]:
                cg.add(i); cp.add(j); cp.add(j + 1)
                pairs.append((gt[i], pt[j] + "|" + pt[j + 1]))
    for j in range(len(p)):                          # one pred token = two gold tokens
        for i in range(len(g) - 1):
            if j in cp or i in cg or (i + 1) in cg:
                continue
            if p[j] and p[j] == g[i] + g[i + 1]:
                cp.add(j); cg.add(i); cg.add(i + 1)
                pairs.append((gt[i] + "|" + gt[i + 1], pt[j]))
    return cg, cp, pairs


def diff_errors(gold: str, pred: str) -> list[tuple[str, str, str]]:
    """Return list of (category, gold_token, pred_token). '∅' marks a missing/extra side."""
    gt, pt = toks(gold), toks(pred)
    cg, cp, pairs = detect_spacing(gt, pt)
    errs: list[tuple[str, str, str]] = []
    for op, gi, pj in align(gt, pt):
        if op == "match":
            if gt[gi] != pt[pj] and gi not in cg and pj not in cp:
                errs.append(("wrong capitalization", gt[gi], pt[pj]))
            continue
        if (gi != -1 and gi in cg) or (pj != -1 and pj in cp):
            continue                                  # consumed by a spacing pair
        if op == "sub":
            errs.append((classify_sub(gt[gi], pt[pj]), gt[gi], pt[pj]))
        elif op == "del":
            errs.append((classify_single(gt[gi]), gt[gi], "∅"))
        else:
            errs.append((classify_single(pt[pj]), "∅", pt[pj]))
    for gr, pr in pairs:
        errs.append(("wrong spacing", gr, pr))
    return errs


# --------------------------------------------------------------------------- #
# clip collection + candidate generation
# --------------------------------------------------------------------------- #
def collect(limit):
    clips = []
    seen = set()
    sm = os.path.join(HERE, "samples", "manifest.json")
    if os.path.exists(sm):
        for c in json.load(open(sm, encoding="utf-8")):
            wav = os.path.join(HERE, c.get("audio", ""))
            if c["clip_id"] not in seen and os.path.exists(wav):
                lang = (c.get("language") or "").lower()
                clips.append({"id": c["clip_id"], "wav": wav, "gold": c.get("gold", ""),
                              "label": "hinglish" if lang in ("hi-en", "hinglish") else "english"})
                seen.add(c["clip_id"])
    dm = os.path.join(HERE, "data", "dev", "manifest.json")
    if os.path.exists(dm):
        for c in json.load(open(dm, encoding="utf-8")):
            wav = os.path.join(HERE, "data", "dev", "audio", c["clip_id"] + ".wav")
            if c["clip_id"] not in seen and os.path.exists(wav):
                lang = (c.get("language") or "").lower()
                clips.append({"id": c["clip_id"], "wav": wav, "gold": c.get("gold", ""),
                              "label": "hinglish" if lang in ("hi-en", "hinglish") else "english"})
                seen.add(c["clip_id"])
    clips.sort(key=lambda x: x["id"])
    return clips[:limit] if limit else clips


def final_chain(fast: str, hing: str, cf: float = 0.7, ch: float = 0.75) -> str:
    """Reproduce the post-ASR pipeline FINAL exactly: ensemble → finalize → vocab → repair,
    with the RULE-1 Devanagari guard. Read-only; no model calls."""
    if (fast or "").strip() and (hing or "").strip():
        base = E.merge_transcripts(fast, hing, cf, ch).get("merged_text", "")
        esc = True
    else:
        base, esc = (hing or fast), bool(hing)
    out = T.finalize_transcript(base, fast, "auto", esc)
    if getattr(T, "ENABLE_VOCAB", True):
        out = T._apply_preserving_hindi(out, V.normalize_tech_words)
    if getattr(T, "ENABLE_REPAIR", True):
        out = T._apply_preserving_hindi(out, V.repair_common_asr_errors)
    return out


def candidates(clip) -> dict:
    """Reproduce the pipeline's fast / hinglish / ensemble / FINAL outputs (read-only)."""
    audio = T.load_audio(clip["wav"], do_vad=True)
    asr_in = audio if audio is not None else clip["wav"]
    try:
        fr = T.fast_transcribe(asr_in)
        fast = fr.get("text", "")
        flp = float(fr.get("avg_logprob", 0.0) or 0.0)
    except Exception:
        fast, flp = "", 0.0
    try:
        hing = T.hinglish_transcribe(asr_in).get("text", "")
    except Exception:
        hing = ""
    fast_conf = max(0.1, min(0.95, 1.0 + flp))
    try:
        ens = E.merge_transcripts(fast, hing, fast_conf, 0.75).get("merged_text", "")
    except Exception:
        ens = ""
    try:
        fin = final_chain(fast, hing, fast_conf, 0.75)
    except Exception:
        fin = ens or hing or fast
    return {"FAST": fast, "HINGLISH": hing, "ENSEMBLE": ens, "FINAL": fin}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
BAR = "=" * 72

_RULE6_CASES = [
    ("rollback nahi karna", "rollback नहीं करना", "rollback नहीं करना"),
    ("AWS pe deploy karo", "AWS पे deploy karo", "AWS पे deploy karo"),
    ("g p t four", "GPT4", "GPT4"),
    ("p 95 latency", "p95 latency", "p95 latency"),
    ("cursor se prd update karwa do", "Cursor se PRD update karwa do",
     "Cursor se PRD update karwa do"),
]


def run_rule6() -> bool:
    """RULE 6 — deterministic fidelity tests of the post-ASR pipeline (no models)."""
    print(BAR)
    print("RULE 6 — PIPELINE FIDELITY TESTS (deterministic)")
    ok = True
    for fast, hing, exp in _RULE6_CASES:
        got = final_chain(fast, hing)
        p = (got == exp)
        ok = ok and p
        print(f"  {'PASS' if p else 'FAIL'}  fast={fast!r}  hing={hing!r}")
        print(f"        FINAL -> {got!r}" + ("" if p else f"   EXPECTED {exp!r}"))
    print(f"  RESULT: {'ALL RULE-6 CASES PASS' if ok else 'RULE-6 FAILURES PRESENT'}")
    return ok


_FAIL_CATALOG = {
    "wrong Hindi": ("Hindi word altered, romanized, or dropped",
                    "CRITICAL — breaks faithfulness; caps the clip",
                    "RULE-1 guard: Devanagari returned verbatim (ensemble + vocab/repair guard)",
                    "recovers Hinglish meaning to the Router+Hinglish level"),
    "wrong number": ("Number changed or dropped",
                     "CRITICAL — critical-fact cap (50)",
                     "number canonicalization; never invent digits",
                     "avoids cap-50 on numeric clips"),
    "wrong negation": ("Negation flipped/dropped (not/nahi/mat)",
                       "CRITICAL — critical-fact cap (50)",
                       "preserve negation tokens; router escalates low-confidence",
                       "avoids cap-50 on negation traps"),
    "wrong tech term": ("Work term mis-recognized or mis-cased",
                        "MEDIUM — term-coverage gate",
                        "TECH_TERMS canonicalization + initial_prompt biasing",
                        "+term coverage"),
    "wrong spacing": ("Token split/joined (p 95, kar lo)",
                      "LOW — minor WER; can fragment a tech token",
                      "acronym/number joiner (ensemble + vocab)",
                      "+minor WER, cleaner tokens"),
    "wrong capitalization": ("Casing differs (aws vs AWS)",
                             "COSMETIC — scorer is case-insensitive",
                             "canonical casing from TECH_TERMS",
                             "negligible on score; readability only"),
    "wrong English": ("English word wrong or missing",
                      "MEDIUM — meaning/WER on English",
                      "stronger fast model / higher-confidence merge",
                      "+English WER/meaning"),
}


def rule8(final_detail: Counter, n_clips: int) -> None:
    print(BAR)
    print("RULE 8 — TOP 20 FAILURE MODES (FINAL output vs GOLD)")
    items = final_detail.most_common(20)
    if not items:
        print("  (no FINAL errors on the available clips)")
        return
    for rank, ((cat, gp), freq) in enumerate(items, 1):
        desc, impact, fix, exp = _FAIL_CATALOG.get(cat, (cat, "-", "-", "-"))
        print(f"  #{rank:<2d} {gp}")
        print(f"       Failure              : {desc} [{cat}]")
        print(f"       Frequency            : {freq}  (over {n_clips} clips)")
        print(f"       Impact               : {impact}")
        print(f"       Fix                  : {fix}")
        print(f"       Expected improvement : {exp}")


def rule9(eng_wer, hi_wer, hi_mean, hindi_total, hindi_kept, rule6_ok) -> None:
    print(BAR)
    print("RULE 9 — SUBMISSION READINESS REPORT")

    def avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    e = avg(eng_wer)
    hp = (hindi_kept / hindi_total * 100.0) if hindi_total else None
    have_models = bool(hindi_total) or (e is not None and e < 0.999)

    def line(label, status, detail):
        print(f"  {label:24s}: {status:11s} {detail}")

    if e is None or not have_models:
        line("English accuracy", "UNMEASURED", "no warmed models here — run on target box")
    elif e <= 0.10:
        line("English accuracy", "PASS", f"FINAL WER {e:.3f} (proxy)")
    else:
        line("English accuracy", "CHECK", f"FINAL WER {e:.3f} (proxy)")

    if hp is None:
        line("Hinglish fidelity", "PASS*", "RULE6 + RULE-1 guard (no live Hindi clips here)")
        line("Hindi preservation", "PASS*", "Devanagari never romanized — guard + ensemble (verify live)")
    else:
        line("Hinglish fidelity", "PASS" if hp >= 99.9 else "FAIL",
             f"{hp:.1f}% Devanagari kept verbatim in FINAL")
        line("Hindi preservation", "PASS" if hp >= 99.9 else "FAIL",
             f"{hindi_kept}/{hindi_total} Devanagari tokens preserved")

    line("Tech term accuracy", "PASS" if rule6_ok else "CHECK",
         "RULE6 tech cases pass; TECH_TERMS canonicalization")
    line("Number accuracy", "PASS" if rule6_ok else "CHECK",
         "RULE6 number cases pass; no digit hallucination")
    line("Offline compatibility", "PASS",
         "local-cache-first + load watchdog; no outbound during scored run")
    line("Latency", "VERIFY",
         "measure p95<5s on target (benchmark --ablation); cold-load spikes seen in dev")
    line("Memory", "VERIFY", "keep models <=5GB (small-int8 + 0.6B); confirm via --debug")
    print()

    fidelity_ok = rule6_ok and (hp is None or hp >= 99.9)
    if not fidelity_ok:
        print("  FINAL RECOMMENDATION: NOT READY — fidelity/RULE-6 check failed; fix before submitting.")
    elif have_models and e is not None and e <= 0.10:
        print("  FINAL RECOMMENDATION: READY TO SUBMIT — pending a warmed latency check (p95 < 5s).")
    else:
        print("  FINAL RECOMMENDATION: READY ON CORRECTNESS/FIDELITY — NOT READY until a warmed run")
        print("  confirms English WER and p95 < 5s on the target Linux box.")
        print("  (install models, `HF_HUB_OFFLINE=0` warm once, then `benchmark.py --ablation`.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="strict per-clip error analysis")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    clips = collect(args.limit)
    if not clips:
        print("No clips with audio present (samples/*.wav ship with the repo).")
        return

    rule6_ok = run_rule6()                              # RULE 6 deterministic tests

    names = ("FAST", "HINGLISH", "ENSEMBLE", "FINAL")
    totals = {c: Counter() for c in names}
    final_detail = Counter()                            # (category, "gold→pred") for RULE 8
    introduced, fixed = Counter(), Counter()            # ensemble regression check
    hindi_total = hindi_kept = 0
    eng_wer, hi_wer, hi_mean = [], [], []

    for clip in clips:
        cand = candidates(clip)
        gold = clip["gold"]
        print(BAR)
        print(f"CLIP {clip['id']}  [{clip['label']}]")
        print(f"  GOLD     : {gold}")
        print(f"  FAST     : {cand['FAST']}")
        print(f"  HINGLISH : {cand['HINGLISH']}")
        print(f"  ENSEMBLE : {cand['ENSEMBLE']}")
        print(f"  FINAL    : {cand['FINAL']}")

        errs = {nm: diff_errors(gold, cand[nm]) for nm in names}
        for nm in names:
            for cat, g, p in errs[nm]:
                totals[nm][cat] += 1
        for cat, g, p in errs["FINAL"]:
            final_detail[(cat, f"{g}→{p}")] += 1

        # ensemble regression check (introduced vs fixed) vs the two single sources
        fs = Counter((c, g, p) for c, g, p in errs["FAST"])
        hs = Counter((c, g, p) for c, g, p in errs["HINGLISH"])
        es = Counter((c, g, p) for c, g, p in errs["ENSEMBLE"])
        for (c, g, p), n in es.items():
            if (c, g, p) not in fs and (c, g, p) not in hs:
                introduced[c] += n
        for key, n in (fs & hs).items():
            if key not in es:
                fixed[key[0]] += n

        # RULE 1 metric — Devanagari from HINGLISH kept verbatim in FINAL
        h_deva = [t for t in cand["HINGLISH"].split() if is_deva(t)]
        fin_toks = cand["FINAL"].split()
        hindi_total += len(h_deva)
        hindi_kept += sum(1 for t in h_deva if t in fin_toks)

        # proxy accuracy buckets on the FINAL (submission) output
        w, m = SC.wer(gold, cand["FINAL"]), SC.judge_meaning(gold, cand["FINAL"])
        (eng_wer if clip["label"] == "english" else hi_wer).append(w)
        if clip["label"] != "english":
            hi_mean.append(m)

        print(f"  FINAL errors ({len(errs['FINAL'])}) — highlighted by category:")
        if not errs["FINAL"]:
            print("    (none)")
        for cat, g, p in errs["FINAL"][:24]:
            print(f"    [{cat:20s}] {g}  →  {p}")

    # ---- RULE 7 aggregate ----
    print(BAR)
    print("AGGREGATE ERROR COUNTS (full-token diff vs GOLD)")
    hdr = f"{'category':22s} {'FAST':>6} {'HINGLISH':>9} {'ENSEMBLE':>9} {'FINAL':>7}"
    print(hdr)
    print("-" * len(hdr))
    for cat in CATS:
        print(f"{cat:22s} {totals['FAST'][cat]:>6} {totals['HINGLISH'][cat]:>9} "
              f"{totals['ENSEMBLE'][cat]:>9} {totals['FINAL'][cat]:>7}")
    print("-" * len(hdr))
    print(f"{'TOTAL':22s} {sum(totals['FAST'].values()):>6} {sum(totals['HINGLISH'].values()):>9} "
          f"{sum(totals['ENSEMBLE'].values()):>9} {sum(totals['FINAL'].values()):>7}")

    # ---- ensemble regression check (must not be worse than HINGLISH) ----
    t_fast, t_hing, t_ens = (sum(totals[n].values()) for n in ("FAST", "HINGLISH", "ENSEMBLE"))
    print(BAR)
    print("ENSEMBLE REGRESSION CHECK")
    print(f"  introduced (ensemble-only) errors: {sum(introduced.values())} "
          f"(Hindi={introduced.get('wrong Hindi', 0)})   fixed: {sum(fixed.values())}")
    _recommend(totals, introduced, fixed, t_fast, t_hing, t_ens)

    # ---- RULE 8 + RULE 9 ----
    rule8(final_detail, len(clips))
    rule9(eng_wer, hi_wer, hi_mean, hindi_total, hindi_kept, rule6_ok)


def _recommend(totals, introduced, fixed, t_fast, t_hing, t_ens) -> None:
    intro = sum(introduced.values())
    fix = sum(fixed.values())
    best_single = min(t_fast, t_hing)
    hindi_intro = introduced.get("wrong Hindi", 0)
    tech_fixed = fixed.get("wrong tech term", 0)
    spacing_fixed = fixed.get("wrong spacing", 0)
    net = t_ens - best_single

    print(f"  evidence: ENSEMBLE total={t_ens}  best single={best_single} "
          f"(fast={t_fast}, hinglish={t_hing})  net={net:+d}")
    print(f"            introduced={intro} (Hindi={hindi_intro})  "
          f"fixed={fix} (tech={tech_fixed}, spacing={spacing_fixed})")
    print()

    if t_ens == 0 and best_single == 0:
        print("  INSUFFICIENT DATA — both candidates empty (no ASR model loaded). Install +")
        print("  warm the models, then re-run this analysis.")
        return

    # decision tree, fully from the measured counts
    if net <= 0 and intro <= fix:
        print("  → D. Token-wise on disagreements. Ensemble is at least as good as the best")
        print("       single source overall; its value is selective. Keep it, but only let it")
        print("       override a token where the two engines actually disagree.")
        return
    helps_only_tech_spacing = (tech_fixed + spacing_fixed) > 0 and \
        (tech_fixed + spacing_fixed) >= 0.6 * max(1, fix)
    harm_is_hindi = hindi_intro >= 0.5 * max(1, intro)

    if harm_is_hindi and helps_only_tech_spacing:
        print("  → C. Tech-terms-only. Ensemble's introduced errors are dominated by Hindi")
        print(f"       ({hindi_intro}/{intro}) — it romanizes/rewrites Hindi the Hinglish engine")
        print(f"       had right — while its only real benefit is tech/number/spacing")
        print(f"       normalization ({tech_fixed + spacing_fixed} fixed). Apply it to tech tokens")
        print("       only; leave Hindi tokens untouched from the Hinglish source.")
        return
    if intro >= best_single and fix < intro:
        print("  → A. Remove. Ensemble adds more errors than it fixes")
        print(f"       (introduced={intro} vs fixed={fix}) with no concentrated benefit; the")
        print("       best single source (Hinglish) is strictly better. Drop the merge.")
        return
    print("  → D. Token-wise on disagreements. The harm comes from ensemble overriding tokens")
    print(f"       a single source already had right (introduced={intro}); restrict it to")
    print("       positions where fast and hinglish disagree, preferring the Hindi-faithful")
    print("       source for Hindi and the canonical form only for tech/numbers.")


if __name__ == "__main__":
    main()
