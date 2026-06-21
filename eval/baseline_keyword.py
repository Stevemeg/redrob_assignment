"""
Baseline 1: pure keyword matching.

Counts how many of the JD's literal "must-have" skill tokens appear
verbatim in a candidate's skill list, with zero semantic understanding --
this is what most naive hackathon submissions (and the bundled
sample_submission.csv, which ranks candidates by raw "AI core skills"
count) actually do. We implement it faithfully as a baseline specifically
so its weaknesses are visible in the eval comparison, not to use it for
real ranking.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import jd_profile as jd


ALL_MUST_HAVE_TOKENS = set()
for tokens in jd.MUST_HAVE_SKILL_GROUPS.values():
    ALL_MUST_HAVE_TOKENS |= tokens


def keyword_match_score(candidate: dict) -> float:
    skill_names = {s["name"].strip().lower() for s in candidate.get("skills", [])}
    hits = len(skill_names & ALL_MUST_HAVE_TOKENS)
    # Normalize by total possible distinct tokens actually present in the
    # vocabulary (so score is comparable/bounded), then scale by raw count
    # the way a naive "count AI keywords" approach would.
    return hits / max(1, len(ALL_MUST_HAVE_TOKENS))
