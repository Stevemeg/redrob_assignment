"""
Proxy relevance ground truth for offline evaluation.

We do NOT have the hidden ground truth the organizers use to score real
submissions -- that's only revealed after the competition closes. To still
do rigorous offline evaluation (and to have *something* to report a
measurable-improvement number against, as any real ranking project should),
we build a transparent, rule-based proxy relevance tier using signals that
are clearly and deliberately built into the dataset:

  Tier 4 (highly relevant): core-AI title AND meaningful skill-cluster
          coverage AND no credibility flags.
  Tier 3 (relevant):        adjacent-engineering title with strong skill-
          cluster coverage, OR core-AI title with partial coverage --
          i.e. plausible Tier-5-style candidates and solid-but-imperfect
          core candidates.
  Tier 1 (marginal):        some skill cluster presence but title/career
          history doesn't support it (the "keyword stuffer" pattern).
  Tier 0 (not relevant):    no meaningful skill-cluster presence, OR any
          credibility flag triggered (these are exactly the documented
          honeypots, which the spec says are FORCED to tier 0 in the real
          ground truth -- this part of our proxy is not a guess, it's
          stated organizer methodology).

This proxy is deliberately built from DIFFERENT logic paths than scorer.py
uses for ranking, even though it draws on the same underlying signals --
it's a sanity check on the ranker's behavior, not a circular restatement of
it. A ranker that was overfit to its own scoring logic would not
necessarily score well against this independently-constructed tiering.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skill_ontology import jd_group_coverage
import jd_profile as jd
from credibility import check_credibility


def proxy_relevance_tier(candidate: dict) -> int:
    cred = check_credibility(candidate)
    if cred.any_flag:
        return 0

    skill_names = [s["name"] for s in candidate.get("skills", [])]
    title = (candidate.get("profile", {}).get("current_title") or "").strip().lower()

    coverages = []
    for tokens in jd.MUST_HAVE_SKILL_GROUPS.values():
        cov, _ = jd_group_coverage(skill_names, tokens)
        coverages.append(cov)
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    is_core_title = title in jd.CORE_AI_TITLES
    is_adjacent_title = title in jd.ADJACENT_ENGINEERING_TITLES

    if is_core_title and avg_coverage >= 0.5:
        return 4
    if (is_core_title and avg_coverage >= 0.2) or (is_adjacent_title and avg_coverage >= 0.6):
        return 3
    # Any nonzero skill-cluster presence at all is "marginal", not
    # "irrelevant" -- a candidate touching even one relevant cluster
    # (e.g. lists Pinecone or Haystack once) is a real, if weak, signal and
    # should not be lumped in with profiles that have zero overlap. The
    # original >= 0.2 threshold here created a labeling gap: candidates
    # with avg_coverage in (0, 0.2) fell through to tier 0 -- indistinguishable
    # from true non-fits and honeypots -- even though they're not flagged
    # by check_credibility and clearly have *some* relevant skill evidence.
    if avg_coverage > 0.0:
        return 1
    return 0
