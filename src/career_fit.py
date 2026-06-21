"""
Career-trajectory and contextual-fit checks: the JD's "things we explicitly
do NOT want" section, plus location fit.

Each check returns a (multiplier, note_or_None) pair so they compose
multiplicatively and each one's contribution is traceable in the final
reasoning string -- this matters for Stage 4 manual review, which
specifically checks whether reasoning "acknowledges obvious gaps or
concerns" rather than only ever being positive.
"""

from __future__ import annotations

import jd_profile as jd


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def title_chaser_penalty(career_history: list[dict]) -> tuple[float, str | None]:
    """
    JD: "If your career trajectory shows you optimizing for 'Senior' ->
    'Staff' -> 'Principal' titles by switching companies every 1.5 years,
    we're not a fit."

    Heuristic: 3+ jobs, all individually <= 20 months, with no single
    tenure exceeding ~2.5 years -- i.e., a consistent pattern of short
    stays, not just one short stint.
    """
    if len(career_history) < 3:
        return 1.0, None
    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    if all(d <= 20 for d in durations) and max(durations) <= 24:
        avg = sum(durations) / len(durations)
        return 0.80, f"career pattern shows {len(durations)} roles averaging {avg:.0f} months each (frequent switching)"
    return 1.0, None


def consulting_only_penalty(career_history: list[dict], current_company: str) -> tuple[float, str | None]:
    """
    JD: "People who have only worked at consulting firms... in their entire
    career [are not a fit]. If you're currently at one of these but have
    prior product-company experience, that's fine."
    """
    companies = [_norm(j.get("company", "")) for j in career_history] + [_norm(current_company)]
    if not companies:
        return 1.0, None
    consulting_hits = sum(1 for c in companies if any(cf in c for cf in jd.CONSULTING_FIRMS))
    if consulting_hits == len(companies) and consulting_hits > 0:
        return 0.45, "entire career history is at consulting/IT-services firms with no product-company experience"
    return 1.0, None


def location_fit(location: str, country: str, willing_to_relocate: bool) -> tuple[float, str | None]:
    """
    JD: Pune/Noida preferred, Hyderabad/Pune/Mumbai/Delhi-NCR welcome,
    outside-India is case-by-case with no visa sponsorship.
    """
    loc = _norm(location)
    if country != "India":
        if willing_to_relocate:
            return 0.55, f"based outside India ({location}, {country}) but open to relocation; no visa sponsorship offered"
        return 0.20, f"based outside India ({location}, {country}), not open to relocation, and role does not sponsor visas"

    if any(p in loc for p in jd.PREFERRED_LOCATIONS):
        return 1.05, f"based in {location}, a JD-preferred location"
    if any(w in loc for w in jd.WELCOME_LOCATIONS):
        return 1.0, None
    if willing_to_relocate:
        return 0.90, f"based in {location}, not a listed preferred city, but open to relocation"
    return 0.75, f"based in {location}, not a listed preferred city, and not open to relocation"


def notice_period_fit_note(notice_days: int) -> str | None:
    """Pure annotation helper; actual scoring lives in behavioral_signals."""
    if notice_days is None:
        return None
    if notice_days > 60:
        return f"notice period of {notice_days} days is on the long side for this role"
    return None
