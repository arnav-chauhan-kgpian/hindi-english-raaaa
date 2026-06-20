"""Build the challenge dev/hidden manifests from the validated RambleFix public pool.

Each pool row carries provenance (dataset_repo / config / split / id) + gold + terms,
so the GitHub Action can re-fetch the exact audio from HuggingFace by id — no private
audio needed, fully reproducible online.

  python scripts/build_corpus.py [--pool /path/to/public_launch_dictation_pool.json]

Writes:
  data/dev/manifest.json     public sample (gold shipped — for local preview)
  data/hidden/manifest.json  held-out (gitignored — gold NEVER published)

Split is stratified by category and deterministic (sorted by id), so dev and hidden
never share a clip and the same input always yields the same split.
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
# Point at the RambleFix public pool via env var (keeps personal paths out of the repo):
#   RAMBLEFIX_POOL=/path/to/public_launch_dictation_pool_20260613.json python scripts/build_corpus.py
DEFAULT_POOL = os.environ.get("RAMBLEFIX_POOL", str(HERE / "data/source_pool.json"))
DEV_EVERY = 5  # ~1 in 5 clips per category goes to the public dev set


def to_rec(row: dict, root: Path) -> dict:
    return {
        "clip_id": row["id"],
        "gold": row.get("gold", ""),
        "must_have": row.get("terms", []) or [],
        "language": row.get("language", ""),
        "category": row.get("category", ""),
        "trust": row.get("reference_trust", "silver"),
        # provenance → fetch_audio.py re-fetches the exact clip from HF by id into
        # data/<split>/audio/<id>.wav (no personal paths committed to the repo)
        "audio_ref": {
            "repo": row.get("dataset_repo", ""),
            "config": row.get("dataset_config", ""),
            "split": row.get("dataset_split", ""),
            "id": row["id"],
            "source_audio": os.path.basename(row.get("audio", "")),  # filename only — no local paths
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    args = ap.parse_args()
    pool = json.load(open(args.pool))
    root = Path(args.pool).resolve().parent.parent  # eval_corpus/ -> project root

    by_cat = defaultdict(list)
    for row in pool:
        by_cat[row.get("category", "?")].append(row)

    dev, hidden = [], []
    for cat, rows in by_cat.items():
        rows = sorted(rows, key=lambda r: r["id"])
        for i, row in enumerate(rows):
            (dev if i % DEV_EVERY == 0 else hidden).append(to_rec(row, root))

    (HERE / "data/dev").mkdir(parents=True, exist_ok=True)
    (HERE / "data/hidden").mkdir(parents=True, exist_ok=True)
    json.dump(dev, open(HERE / "data/dev/manifest.json", "w"), indent=2, ensure_ascii=False)
    json.dump(hidden, open(HERE / "data/hidden/manifest.json", "w"), indent=2, ensure_ascii=False)

    res = {c: sum(1 for r in dev if r["category"] == c) for c in by_cat}
    print(f"dev: {len(dev)}  hidden: {len(hidden)}  (total {len(pool)})")
    print("by category (dev/total):")
    for c, rows in sorted(by_cat.items()):
        print(f"  {c:24s} {res.get(c,0):2d} / {len(rows)}")
    print(f"audio fetched per-clip by fetch_audio.py from provenance (HF id / source pool).")


if __name__ == "__main__":
    main()
