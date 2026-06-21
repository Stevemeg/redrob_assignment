"""
Main candidate scoring engine.

Score composition (weights chosen to mirror how the JD itself weighs
things -- see ARCHITECTURE.md for the full rationale):

    raw_fit_score =
          0.42 * skill_match_score        (cluster-adjacency aware)
        + 0.30 * title_seniority_score
        + 0.13 * experience_band_score
        + 0.10 * career_fit_score          (title-chaser / consulting-only / location)
        + 0.05 * education_modifier (as additive bonus, not full weight slot)

    final_score = raw_fit_score * behavioral_multiplier * credibility_multiplier

Credibility flags act as a steep multiplicative penalty (not a hard
disqualifier, per the JD's own instruction not to special-case honeypots --
a system that *naturally* downranks internally-contradictory profiles is
exactly what's being asked for).

Every component records a human-readable note; reason_string() assembles
these into the 1-2 sentence justification the submission CSV requires,
always grounded in fields that actually exist on the candidate (never
inventing skills/employers not present in the profile -- Stage 4 review
explicitly penalizes hallucinated reasoning).
"""

from __future__ import annotations
from dataclasses import dataclass, field

import jd_profile as jd
from skill_ontology import jd_group_coverage, is_plain_language_tier5_skill
from title_classifier import classify_title, career_history_title_boost
from behavioral_signals import score_behavioral_signals
from credibility import check_credibility
from career_fit import title_chaser_penalty, consulting_only_penalty, location_fit
from experience_education import experience_band_score, education_modifier


SKILL_GROUP_WEIGHTS = {
    "embeddings_retrieval": 0.40,
    "vector_db_hybrid_search": 0.25,
    "python_engineering": 0.15,
    "eval_frameworks": 0.20,
}

NICE_TO_HAVE_BONUS_CAP = 0.08  # max additive bonus from "nice to have" skills

COMPONENT_WEIGHTS = {
    "skill_match": 0.42,
    "title": 0.30,
    "experience_band": 0.13,
    "career_fit": 0.10,
}
EDUCATION_BONUS_WEIGHT = 0.05  # additive on top, capped


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    credibility_flagged: bool = False
    is_plain_language_profile: bool = False
    summary_facts: dict = field(default_factory=dict)


def _skill_match(skills: list[dict]) -> tuple[float, list[str]]:
    skill_names = [s["name"] for s in skills]
    notes = []

    weighted_sum = 0.0
    matched_anything: set[str] = set()
    for group, weight in SKILL_GROUP_WEIGHTS.items():
        tokens = jd.MUST_HAVE_SKILL_GROUPS[group]
        cov, matched = jd_group_coverage(skill_names, tokens)
        weighted_sum += weight * cov
        matched_anything |= set(matched)

    base_score = weighted_sum  # weights sum to 1.0 -> already normalized

    # nice-to-have bonus
    nice_bonus = 0.0
    for group, tokens in jd.NICE_TO_HAVE_SKILL_GROUPS.items():
        cov, _ = jd_group_coverage(skill_names, tokens)
        nice_bonus += cov
    nice_bonus = min(NICE_TO_HAVE_BONUS_CAP, nice_bonus * 0.04)

    total = min(1.0, base_score + nice_bonus)

    plain_lang_matches = [s for s in skill_names if is_plain_language_tier5_skill(s)]
    if plain_lang_matches:
        notes.append(
            f"uses plain-language descriptions ({', '.join(plain_lang_matches[:3])}) rather than "
            "buzzwords for what is substantively the same retrieval/ranking expertise"
        )
    elif matched_anything:
        shown = sorted(matched_anything)[:5]
        notes.append(f"core skill match on {', '.join(shown)}")
    else:
        notes.append("no meaningful overlap with the embeddings/retrieval/vector-DB skill set the JD requires")

    return total, notes


def score_candidate(candidate: dict) -> ScoredCandidate:
    cid = candidate["candidate_id"]
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    notes: list[str] = []
    components: dict = {}

    # --- Skill match ---------------------------------------------------
    skill_score, skill_notes = _skill_match(skills)
    components["skill_match"] = skill_score
    notes.extend(skill_notes)

    # --- Title / seniority -----------------------------------------------
    title_assessment = classify_title(profile.get("current_title", ""))
    title_score = title_assessment.title_multiplier
    boost, boost_note = career_history_title_boost(career_history)
    title_score = min(1.0, title_score + boost)
    components["title"] = title_score
    notes.append(title_assessment.explanation)
    if boost_note:
        notes.append(boost_note)

    # --- Experience band ---------------------------------------------------
    exp_score, exp_note = experience_band_score(profile.get("years_of_experience", 0))
    components["experience_band"] = exp_score
    if exp_note:
        notes.append(exp_note)

    # --- Career fit (title-chaser, consulting-only, location) --------------
    career_fit_mult = 1.0
    tc_mult, tc_note = title_chaser_penalty(career_history)
    career_fit_mult *= tc_mult
    if tc_note:
        notes.append(tc_note)

    co_mult, co_note = consulting_only_penalty(career_history, profile.get("current_company", ""))
    career_fit_mult *= co_mult
    if co_note:
        notes.append(co_note)

    loc_mult, loc_note = location_fit(
        profile.get("location", ""), profile.get("country", ""),
        signals.get("willing_to_relocate", False),
    )
    career_fit_mult *= loc_mult
    if loc_note:
        notes.append(loc_note)

    career_fit_mult = max(0.15, min(1.10, career_fit_mult))
    components["career_fit"] = career_fit_mult

    # --- Education bonus -------------------------------------------------
    edu_mult = education_modifier(education)

    # --- Raw fit score (weighted composite) ---------------------------------
    raw_fit = (
        COMPONENT_WEIGHTS["skill_match"] * components["skill_match"]
        + COMPONENT_WEIGHTS["title"] * components["title"]
        + COMPONENT_WEIGHTS["experience_band"] * components["experience_band"]
        + COMPONENT_WEIGHTS["career_fit"] * components["career_fit"]
    )
    raw_fit *= (1.0 + EDUCATION_BONUS_WEIGHT * (edu_mult - 1.0))

    # --- Behavioral signal multiplier ---------------------------------------
    behavior = score_behavioral_signals(signals)
    notes.extend(behavior.notes)

    # --- Credibility check ---------------------------------------------------
    cred = check_credibility(candidate)
    credibility_mult = 1.0
    if cred.any_flag:
        # Each independent flag compounds the penalty; this is intentionally
        # severe since these patterns indicate the profile may not be real.
        credibility_mult = 0.15 ** cred.flag_count
        notes.extend([f"profile credibility concern: {r}" for r in cred.reasons()])

    final_score = raw_fit * behavior.multiplier * credibility_mult
    # NOTE: deliberately not clamped to [0, 1] here. The component weights
    # and bonus multipliers can combine to exceed 1.0 for genuinely
    # excellent candidates (perfect skill+title match plus behavioral and
    # education bonuses) -- clamping at this stage would flatten distinct
    # top candidates to an identical ceiling value, destroying exactly the
    # ranking signal that matters most for NDCG@10 (50% of the composite
    # score). Final CSV-facing scores are rescaled into [0, 1] once, after
    # all 100k candidates are scored, by rank.py -- see normalize_scores().
    final_score = max(0.0, final_score)

    return ScoredCandidate(
        candidate_id=cid,
        score=final_score,
        components={**components, "behavioral_multiplier": behavior.multiplier,
                    "credibility_multiplier": credibility_mult, "education_modifier": edu_mult},
        notes=notes,
        credibility_flagged=cred.any_flag,
        is_plain_language_profile=any(is_plain_language_tier5_skill(s["name"]) for s in skills),
        summary_facts={
            "current_title": profile.get("current_title"),
            "current_company": profile.get("current_company"),
            "years_of_experience": profile.get("years_of_experience"),
            "location": profile.get("location"),
            "country": profile.get("country"),
            "notice_period_days": signals.get("notice_period_days"),
            "recruiter_response_rate": signals.get("recruiter_response_rate"),
        },
    )
