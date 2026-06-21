"""
Experience-band and education-tier scoring.

Kept deliberately small: the JD itself says years-of-experience is "a range,
not a requirement" and that they'll consider candidates outside the band if
other signals are strong, so this is a soft, smoothly-decaying score rather
than a cutoff -- it should never single-handedly veto a candidate.
"""

from __future__ import annotations

import jd_profile as jd


def experience_band_score(years_of_experience: float) -> tuple[float, str | None]:
    lo, hi = jd.EXPERIENCE_BAND
    if lo <= years_of_experience <= hi:
        return 1.0, None

    if years_of_experience < lo:
        gap = lo - years_of_experience
    else:
        gap = years_of_experience - hi

    # Smooth decay: full credit inside the band, gentle decay for the first
    # ~2.5 years outside it (JD explicitly tolerates this), steeper after.
    margin = jd.EXPERIENCE_SOFT_MARGIN
    if gap <= margin:
        score = 1.0 - 0.25 * (gap / margin)
        return score, None
    extra = gap - margin
    score = max(0.35, 0.75 - 0.12 * extra)
    direction = "below" if years_of_experience < lo else "above"
    return score, f"{years_of_experience:.1f} years of experience is {gap:.1f} years {direction} the JD's 5-9yr band"


# Education is explicitly NOT emphasized in this JD (no degree requirement
# mentioned at all) -- we give a small bonus for tier_1 institutions and
# otherwise stay neutral, rather than building a heavy-weighted component
# the JD never asked for.
_TIER_BONUS = {"tier_1": 1.05, "tier_2": 1.0, "tier_3": 1.0, "tier_4": 0.98, "unknown": 1.0}


def education_modifier(education: list[dict]) -> float:
    if not education:
        return 1.0
    best = 1.0
    for e in education:
        tier = e.get("tier", "unknown")
        best = max(best, _TIER_BONUS.get(tier, 1.0))
    return best
