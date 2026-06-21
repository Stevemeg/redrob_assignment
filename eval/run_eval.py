#!/usr/bin/env python3
"""
Offline evaluation: compares our ranker against two baselines (pure keyword
matching, and TF-IDF cosine similarity) using the same NDCG@10 / NDCG@50 /
MAP / P@10 metrics and weights the real submission is scored on (per
submission_spec.md Section 4), computed against the proxy ground truth in
proxy_ground_truth.py.

This is run on a fixed-size sample of the candidate pool (not the full
100k) purely to keep iteration fast during development -- the final
submission ranking in rank.py always runs over the full pool.

Usage:
    python eval/run_eval.py --candidates <path> --sample-size 5000
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer import score_candidate
from baseline_keyword import keyword_match_score
from baseline_tfidf import tfidf_rank
from proxy_ground_truth import proxy_relevance_tier


def _open_candidates(path: str):
    import gzip
    p = Path(path)
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def load_sample(path: str, n: int) -> list[dict]:
    out = []
    with _open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def ndcg_at_k(ranked_relevances: list[int], k: int) -> float:
    def dcg(rels):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    actual = dcg(ranked_relevances)
    ideal = dcg(sorted(ranked_relevances, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranked_relevances: list[int], relevance_threshold: int = 1) -> float:
    relevant_seen = 0
    precisions = []
    total_relevant = sum(1 for r in ranked_relevances if r >= relevance_threshold)
    if total_relevant == 0:
        return 0.0
    for i, r in enumerate(ranked_relevances):
        if r >= relevance_threshold:
            relevant_seen += 1
            precisions.append(relevant_seen / (i + 1))
    return sum(precisions) / total_relevant


def precision_at_k(ranked_relevances: list[int], k: int, relevance_threshold: int = 3) -> float:
    top_k = ranked_relevances[:k]
    return sum(1 for r in top_k if r >= relevance_threshold) / k


def composite(ranked_relevances: list[int]) -> dict:
    n10 = ndcg_at_k(ranked_relevances, 10)
    n50 = ndcg_at_k(ranked_relevances, 50)
    m = average_precision(ranked_relevances)
    p10 = precision_at_k(ranked_relevances, 10)
    final = 0.50 * n10 + 0.30 * n50 + 0.15 * m + 0.05 * p10
    return {"NDCG@10": n10, "NDCG@50": n50, "MAP": m, "P@10": p10, "composite": final}


def evaluate_system(name: str, candidates: list[dict], scores: list[float], tiers: list[int], top_n: int = 100) -> dict:
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    top = order[:top_n]
    ranked_relevances = [tiers[i] for i in top]
    metrics = composite(ranked_relevances)
    metrics["honeypots_in_top10"] = sum(1 for i in top[:10] if tiers[i] == 0)
    metrics["honeypots_in_top100"] = sum(1 for i in top[:top_n] if tiers[i] == 0)
    metrics["system"] = name
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading {args.sample_size} candidates...", file=sys.stderr)
    candidates = load_sample(args.candidates, args.sample_size)

    print("Computing proxy ground truth tiers...", file=sys.stderr)
    tiers = [proxy_relevance_tier(c) for c in candidates]
    tier_counts = {t: tiers.count(t) for t in sorted(set(tiers))}
    print(f"Tier distribution: {tier_counts}", file=sys.stderr)

    print("Scoring: keyword baseline...", file=sys.stderr)
    kw_scores = [keyword_match_score(c) for c in candidates]

    print("Scoring: TF-IDF baseline...", file=sys.stderr)
    t0 = time.time()
    tfidf_scores = tfidf_rank(candidates)
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr)

    print("Scoring: our ranker...", file=sys.stderr)
    t0 = time.time()
    our_scores = [score_candidate(c).score for c in candidates]
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr)

    results = [
        evaluate_system("Baseline 1: Keyword matching", candidates, kw_scores, tiers),
        evaluate_system("Baseline 2: TF-IDF cosine similarity", candidates, tfidf_scores, tiers),
        evaluate_system("Our ranker (structured + ontology + credibility)", candidates, our_scores, tiers),
    ]

    print()
    print(f"{'System':<55} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>8} {'P@10':>8} {'Composite':>10} {'HP@10':>6} {'HP@100':>7}")
    for r in results:
        print(f"{r['system']:<55} {r['NDCG@10']:>8.4f} {r['NDCG@50']:>8.4f} {r['MAP']:>8.4f} "
              f"{r['P@10']:>8.4f} {r['composite']:>10.4f} {r['honeypots_in_top10']:>6d} {r['honeypots_in_top100']:>7d}")

    out_path = Path(__file__).resolve().parent.parent / "output" / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
