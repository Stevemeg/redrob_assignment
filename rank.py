#!/usr/bin/env python3
"""
Redrob Hackathon -- candidate ranker.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the full candidate pool (JSONL, optionally gzipped), scores every
candidate with the rule-based engine in src/scorer.py, and writes the
top-100 ranking CSV per submission_spec.md Section 2-3:

    candidate_id,rank,score,reasoning

Design goals driven directly by the compute constraints in
submission_spec.md Section 3 (<=5 min wall clock, <=16GB RAM, CPU only,
no network):
  - Single pass, streaming JSON decode -- never materializes more than one
    candidate's parsed object at a time beyond the running top-K heap.
  - No ML model loading, no embedding computation, no network calls.
    Everything is closed-form arithmetic over structured fields, which is
    why this comfortably finishes in seconds rather than minutes even on
    a single core.
  - Memory use is O(K) for the heap of best-so-far candidates, not O(N).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scorer import score_candidate  # noqa: E402
from reasoning import build_reasoning  # noqa: E402


TOP_N = 100


def _open_candidates(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def run(candidates_path: str, out_path: str) -> None:
    t0 = time.time()

    # Min-heap of size TOP_N keyed on score, so we only ever hold TOP_N + 1
    # scored candidates in memory regardless of pool size. Tie-break key
    # mirrors the submission spec's required tie-break (candidate_id
    # ascending) by storing candidate_id as the heap's secondary sort key,
    # negated appropriately so heap semantics give us what we want when we
    # pop everything out at the end and re-sort.
    heap: list[tuple[float, str, object]] = []  # (score, candidate_id, ScoredCandidate)
    n_processed = 0
    n_errors = 0

    with _open_candidates(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                n_errors += 1
                continue

            try:
                result = score_candidate(candidate)
            except Exception:
                # A single malformed record shouldn't crash a 100K-row run.
                # We log and skip; see README for how to surface these.
                n_errors += 1
                continue

            n_processed += 1
            entry = (result.score, result.candidate_id, result)

            if len(heap) < TOP_N:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)

    elapsed = time.time() - t0

    # Sort descending by score; tie-break by candidate_id ascending, exactly
    # matching validate_submission.py's tie-break check.
    top = sorted(heap, key=lambda e: (-e[0], e[1]))

    # Rescale into [0, 1] using a fixed reference ceiling rather than the
    # batch's own max, so scores are stable/interpretable across runs and
    # don't silently change meaning if a future run's top candidate has a
    # different raw value. 1.10 comfortably covers the observed raw range
    # (best real candidates land ~1.00-1.07); anything above is clipped.
    SCORE_CEILING = 1.10
    rescaled_top = [
        (round(min(1.0, score / SCORE_CEILING), 4), cid, result) for (score, cid, result) in top
    ]

    # IMPORTANT: re-sort after rescale+round. Rounding to 4 decimal places
    # can make two previously-distinct raw scores collapse to the same
    # rounded value (this happened in practice: two candidates ~10 ranks
    # apart by raw score rounded to the same 0.7748). The validator checks
    # the tie-break rule against what's actually IN THE CSV (the rounded
    # value), so the final ordering must be computed post-rounding, not
    # inherited from the pre-rounding heap order.
    rescaled_top = sorted(rescaled_top, key=lambda e: (-e[0], e[1]))

    rows = []
    for rank, (score, cid, result) in enumerate(rescaled_top, start=1):
        reasoning = build_reasoning(result, rank)
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    honeypot_in_top100 = sum(1 for _, _, r in top if r.credibility_flagged)

    print(f"Processed {n_processed} candidates ({n_errors} skipped/errored) in {elapsed:.1f}s", file=sys.stderr)
    print(f"Wrote top {len(rows)} to {out_path}", file=sys.stderr)
    print(f"Credibility-flagged candidates in top 100: {honeypot_in_top100}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    run(args.candidates, args.out)


if __name__ == "__main__":
    main()
