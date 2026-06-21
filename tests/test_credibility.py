"""
Unit tests for credibility.py, using minimal synthetic candidate dicts
(not the real dataset) so the tests are self-contained and fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credibility import check_credibility


def _base_candidate(**overrides):
    c = {
        "candidate_id": "CAND_0000000",
        "profile": {"years_of_experience": 6.0},
        "career_history": [
            {"company": "Acme", "title": "Engineer", "duration_months": 72, "is_current": True},
        ],
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 60}],
    }
    c.update(overrides)
    return c


def test_clean_profile_has_no_flags():
    c = _base_candidate()
    flags = check_credibility(c)
    assert not flags.any_flag
    assert flags.flag_count == 0


def test_yoe_mismatch_detected():
    c = _base_candidate(
        profile={"years_of_experience": 13.7},
        career_history=[{"company": "Infosys", "title": "BA", "duration_months": 11, "is_current": True}],
    )
    flags = check_credibility(c)
    assert flags.yoe_mismatch
    assert "13.7" in flags.yoe_mismatch_detail or "0.9" in flags.yoe_mismatch_detail


def test_yoe_within_tolerance_not_flagged():
    # 6.0 yoe = 72 months; career history sums to 70 months -- within the
    # 24-month tolerance, should NOT trigger.
    c = _base_candidate(
        profile={"years_of_experience": 6.0},
        career_history=[{"company": "Acme", "title": "Engineer", "duration_months": 70, "is_current": True}],
    )
    flags = check_credibility(c)
    assert not flags.yoe_mismatch


def test_expert_zero_duration_detected():
    c = _base_candidate(skills=[
        {"name": "MLflow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Photoshop", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Content Writing", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ])
    flags = check_credibility(c)
    assert flags.expert_zero_duration
    assert len(flags.expert_zero_duration_skills) == 3


def test_expert_zero_duration_below_threshold_not_flagged():
    # Only 2 such skills -- below the 3-skill threshold.
    c = _base_candidate(skills=[
        {"name": "MLflow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Photoshop", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ])
    flags = check_credibility(c)
    assert not flags.expert_zero_duration


def test_overqualified_for_experience_detected():
    skills = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 1, "duration_months": 1}
        for i in range(12)
    ]
    c = _base_candidate(profile={"years_of_experience": 1.0}, skills=skills)
    flags = check_credibility(c)
    assert flags.overqualified_for_experience


def test_multiple_flags_compound():
    c = _base_candidate(
        profile={"years_of_experience": 15.0},
        career_history=[{"company": "X", "title": "Y", "duration_months": 6, "is_current": True}],
        skills=[
            {"name": "MLflow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Photoshop", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Content Writing", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
    )
    flags = check_credibility(c)
    assert flags.yoe_mismatch
    assert flags.expert_zero_duration
    assert flags.flag_count == 2


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod) if name.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
