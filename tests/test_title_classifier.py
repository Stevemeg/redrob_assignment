"""Unit tests for title_classifier.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from title_classifier import classify_title, career_history_title_boost


def test_core_ai_title_gets_full_score():
    result = classify_title("Senior AI Engineer")
    assert result.category == "core_ai"
    assert result.title_multiplier == 1.0


def test_non_engineering_title_gets_near_zero():
    result = classify_title("HR Manager")
    assert result.category == "non_engineering"
    assert result.title_multiplier < 0.1


def test_adjacent_engineering_title_gets_partial_credit():
    result = classify_title("Backend Engineer")
    assert result.category == "adjacent_engineering"
    assert 0.3 < result.title_multiplier < 0.8


def test_off_domain_tech_gets_low_but_nonzero():
    result = classify_title("Frontend Engineer")
    assert result.category == "off_domain_tech"
    assert 0.1 < result.title_multiplier < 0.5


def test_case_insensitive():
    a = classify_title("senior ai engineer")
    b = classify_title("SENIOR AI ENGINEER")
    assert a.title_multiplier == b.title_multiplier == 1.0


def test_career_history_boost_for_past_ai_title():
    history = [
        {"title": "Senior Software Engineer"},
        {"title": "ML Engineer"},
    ]
    boost, note = career_history_title_boost(history)
    assert boost > 0
    assert note is not None


def test_career_history_no_boost_when_never_ai_titled():
    history = [{"title": "Business Analyst"}, {"title": "Operations Manager"}]
    boost, note = career_history_title_boost(history)
    assert boost == 0.0
    assert note is None


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
