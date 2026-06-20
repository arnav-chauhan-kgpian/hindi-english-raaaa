"""Proves the scoring engine is fair and hard to game. Run: python tests/test_scorecard.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scorecard import score_clip, wer, critical_flip, has_repetition_loop

GOLD = "Tell Cursor to update the PRD but do not change the June 24 deadline."
GOOD_AUDIT = {"model_ids": ["whisper.cpp-small"], "route": "fast"}
T = {"total": 800}

def clip(pred, must=None, local=True, timings=T, audit=GOOD_AUDIT):
    return score_clip(GOLD, pred, must or ["Cursor", "PRD"], timings, local, audit, "c")

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name

# 1. perfect transcript → high score, no cap
r = clip(GOLD)
check(f"perfect scores high ({r.score})", r.score >= 95 and r.capped_at is None)
check("perfect WER == 0", wer(GOLD, GOLD) == 0.0)

# 2. date/number flip → capped at 50
r = clip("Tell Cursor to update the PRD but do not change the June 21 deadline.")
check(f"date flip capped at 50 ({r.score})", r.capped_at == 50.0 and r.score <= 50)

# 3. negation dropped → critical flip → capped at 50
r = clip("Tell Cursor to update the PRD and change the June 24 deadline.")
check(f"dropped negation capped at 50 ({r.score})", r.capped_at == 50.0)

# 4. required term missing → flip
flipped, reasons = critical_flip(GOLD, "Tell it to update the doc but keep the June 24 date.", ["Cursor", "PRD"])
check("missing required term flagged", flipped)

# 5. blank → capped at 20
r = clip("")
check(f"blank capped at 20 ({r.score})", r.capped_at == 20.0)

# 6. repetition loop → capped at 30
loop = "update the PRD update the PRD update the PRD update the PRD update the PRD"
check("repetition loop detected", has_repetition_loop(loop))
r = clip(loop)
check(f"repetition capped at 30 ({r.score})", r.capped_at == 30.0)

# 7. unrelated output (high WER) → capped at 20
r = clip("the weather in mumbai is hot and humid today honestly")
check(f"unrelated capped at 20 ({r.score})", r.capped_at == 20.0)

# 8. network/local violation → loses local points
r = clip(GOLD, local=False)
check(f"non-local loses 10 pts ({r.score})", r.score < 95 and "local_only flag false" in r.reasons)

print("\nALL SCORECARD TESTS PASSED")
