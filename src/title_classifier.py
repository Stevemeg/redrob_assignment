"""
Title/seniority classification.

This is the single most decisive component against the "keyword stuffer"
trap the JD explicitly calls out: a candidate with a perfect skills list but
the title "Marketing Manager" or "HR Manager" should not rank highly, no
matter how many AI buzzwords are in their skills array. The sample
submission included in the hackathon bundle gets this exactly backwards
(it ranks an "HR Manager with 9 AI core skills" #1) -- this module exists
specifically to not repeat that mistake.

We classify on *current_title* and also scan *career_history* titles, since
a candidate whose current role is adjacent but who previously held a core
AI/ML title (or vice versa -- currently a strong title but with a career
that's mostly non-technical) should be scored on the fuller picture, not
just the most recent title string.
"""

from __future__ import annotations
from dataclasses import dataclass

import jd_profile as jd


def _norm_title(t: str) -> str:
    return (t or "").strip().lower()


@dataclass
class TitleAssessment:
    category: str          # "core_ai", "adjacent_engineering", "off_domain_tech", "non_engineering"
    title_multiplier: float
    explanation: str


def classify_title(title: str) -> TitleAssessment:
    t = _norm_title(title)

    if t in jd.CORE_AI_TITLES:
        return TitleAssessment(
            "core_ai", 1.0,
            f"current title '{title}' is a direct match for the AI/ML engineering role",
        )
    if t in jd.ADJACENT_ENGINEERING_TITLES:
        return TitleAssessment(
            "adjacent_engineering", 0.55,
            f"current title '{title}' is general software/data engineering, not AI-labeled, "
            "but plausibly adjacent depending on actual project history",
        )
    if t in jd.OFF_DOMAIN_TECH_TITLES:
        return TitleAssessment(
            "off_domain_tech", 0.25,
            f"current title '{title}' is technical but in a different discipline "
            "(frontend/mobile/QA/cloud-ops) than NLP/IR/ranking",
        )
    if t in jd.NON_ENGINEERING_TITLES:
        return TitleAssessment(
            "non_engineering", 0.04,
            f"current title '{title}' is non-engineering; AI-sounding skills on a non-engineering "
            "title is the keyword-stuffer pattern the JD explicitly warns about",
        )

    # Unrecognized title string -- treat as weak adjacent rather than zero,
    # since the title vocabulary observed in the data is closed but we
    # shouldn't crash (or zero out) on anything slightly different.
    return TitleAssessment(
        "adjacent_engineering", 0.35,
        f"current title '{title}' not in the recognized vocabulary; scored as weakly adjacent",
    )


def career_history_title_boost(career_history: list[dict]) -> tuple[float, str | None]:
    """
    Look across career history (not just current title) for evidence the
    candidate has held a core-AI title before, even if their current title
    reads as adjacent. Returns (boost, explanation_or_None).

    This protects against penalizing someone who, e.g., stepped back into a
    "Senior Software Engineer" title at a new company but previously held
    "ML Engineer" -- career trajectories aren't always monotonic in title
    text, and the JD itself says it cares about what was actually built.
    """
    best_boost = 0.0
    best_title = None
    for job in career_history:
        t = _norm_title(job.get("title", ""))
        if t in jd.CORE_AI_TITLES:
            best_boost = max(best_boost, 0.20)
            best_title = job.get("title")
    if best_boost > 0:
        return best_boost, f"previously held '{best_title}', evidencing real AI/ML title history"
    return 0.0, None
