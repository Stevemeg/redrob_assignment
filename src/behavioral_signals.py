"""
Behavioral signal scoring, built from the 23 redrob_signals fields.

The JD is explicit about this:
    "a perfect-on-paper candidate who hasn't logged in for 6 months and has
    a 5% recruiter response rate is, for hiring purposes, not actually
    available. Down-weight them appropriately."

This module turns that instruction into a single multiplicative modifier in
[0.3, 1.15] applied on top of the profile-match score, rather than a hard
filter -- a strong-fit candidate who's gone a bit quiet should drop several
ranks, not disappear entirely (we don't actually know *why* they're quiet;
maybe they just accepted an offer elsewhere and haven't updated their flag).

We deliberately keep this as one combined "availability/engagement"
modifier rather than 23 separate weighted terms: most of the 23 fields are
correlated (recruiter_response_rate, avg_response_time_hours,
interview_completion_rate all measure the same underlying "are they
actually reachable and serious" latent factor), and decomposing them
further would just be re-weighting noise. We pick out the handful of
signals with a clear, JD-stated interpretation and ignore the rest as
identity/platform-hygiene fields (verified_email, linkedin_connected, etc.)
that don't bear on fit or availability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime


REFERENCE_DATE = date(2026, 6, 1)  # dataset's generation epoch; last_active_date values cluster near this


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


@dataclass
class BehavioralAssessment:
    multiplier: float
    notes: list[str] = field(default_factory=list)


def score_behavioral_signals(signals: dict) -> BehavioralAssessment:
    notes: list[str] = []
    multiplier = 1.0

    # --- Availability: are they even looking? -----------------------------
    open_to_work = signals.get("open_to_work_flag", False)
    if not open_to_work:
        multiplier *= 0.55
        notes.append("not marked open-to-work")

    # --- Recency of activity ------------------------------------------------
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active is not None:
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive > 180:
            multiplier *= 0.5
            notes.append(f"inactive for {days_inactive} days (>6 months)")
        elif days_inactive > 90:
            multiplier *= 0.75
            notes.append(f"inactive for {days_inactive} days (>3 months)")
        elif days_inactive > 30:
            multiplier *= 0.92
            notes.append(f"last active {days_inactive} days ago")
        # else: recently active, no penalty

    # --- Recruiter responsiveness -------------------------------------------
    response_rate = signals.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate < 0.10:
            multiplier *= 0.55
            notes.append(f"very low recruiter response rate ({response_rate:.0%})")
        elif response_rate < 0.30:
            multiplier *= 0.80
            notes.append(f"low recruiter response rate ({response_rate:.0%})")
        elif response_rate >= 0.70:
            multiplier *= 1.05
            notes.append(f"strong recruiter response rate ({response_rate:.0%})")

    # --- Interview follow-through -------------------------------------------
    interview_completion = signals.get("interview_completion_rate")
    if interview_completion is not None and interview_completion < 0.4:
        multiplier *= 0.85
        notes.append(f"low interview completion rate ({interview_completion:.0%})")

    # --- Notice period (JD: loves sub-30, can buy out up to 30, 30+ "bar gets higher") ---
    notice_days = signals.get("notice_period_days")
    if notice_days is not None:
        if notice_days <= 30:
            multiplier *= 1.05
            notes.append(f"short notice period ({notice_days}d)")
        elif notice_days <= 60:
            pass  # neutral
        else:
            multiplier *= 0.93
            notes.append(f"long notice period ({notice_days}d)")

    # --- GitHub activity: light positive signal only, not a penalty for -1 ---
    gh = signals.get("github_activity_score")
    if gh is not None and gh > 60:
        multiplier *= 1.05
        notes.append(f"strong GitHub activity score ({gh:.0f})")

    multiplier = max(0.3, min(1.15, multiplier))
    return BehavioralAssessment(multiplier=multiplier, notes=notes)
