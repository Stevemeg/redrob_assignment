"""
End-to-end scorer tests against real candidates pulled from the dataset.
These specific candidate_ids were hand-inspected during development (see
ARCHITECTURE.md for the full walkthrough) and represent the four key
archetypes the ranker must get right:

  CAND_0002025 - genuine strong fit: Senior AI Engineer with real
                 production retrieval/ranking/recsys history.
  CAND_0005538 - plain-language Tier-5 candidate: same substance as
                 CAND_0002025 but describes it without AI buzzwords.
  CAND_0000001 - keyword-stuffer trap: Backend/data-engineer with several
                 AI-sounding skills listed but no real AI career history.
  CAND_0003430 - honeypot: internally self-contradictory profile (13.7
                 years claimed, career history sums to 11 months).

This test file requires the real candidates.jsonl to be present (path
configurable via CANDIDATES_PATH env var or the default location used
throughout development); it's skipped gracefully if not found, since CI
environments won't have the 487MB data file.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scorer import score_candidate


DEFAULT_PATH = (
    "/home/claude/work/extracted/[PUB] India_runs_data_and_ai_challenge/"
    "India_runs_data_and_ai_challenge/candidates.jsonl"
)
CANDIDATES_PATH = os.environ.get("CANDIDATES_PATH", DEFAULT_PATH)

TARGET_IDS = {"CAND_0002025", "CAND_0005538", "CAND_0000001", "CAND_0003430"}


def _load_targets():
    found = {}
    if not Path(CANDIDATES_PATH).exists():
        return found
    with open(CANDIDATES_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in TARGET_IDS:
                found[c["candidate_id"]] = c
            if len(found) == len(TARGET_IDS):
                break
    return found


_CANDIDATES = _load_targets()


def test_data_available():
    if not _CANDIDATES:
        print("  (skipped: candidates.jsonl not found at", CANDIDATES_PATH, ")")
        return
    assert len(_CANDIDATES) == len(TARGET_IDS)


def test_strong_fit_scores_highest():
    if not _CANDIDATES:
        return
    scores = {cid: score_candidate(c).score for cid, c in _CANDIDATES.items()}
    assert scores["CAND_0002025"] == max(scores.values())


def test_honeypot_scores_lowest_by_large_margin():
    if not _CANDIDATES:
        return
    scores = {cid: score_candidate(c).score for cid, c in _CANDIDATES.items()}
    assert scores["CAND_0003430"] == min(scores.values())
    # Should be at least 10x lower than the next-lowest, given the
    # credibility multiplier's 0.15^n penalty.
    others = sorted(v for k, v in scores.items() if k != "CAND_0003430")
    assert scores["CAND_0003430"] * 10 < others[0]


def test_plain_language_candidate_beats_keyword_stuffer():
    if not _CANDIDATES:
        return
    r5538 = score_candidate(_CANDIDATES["CAND_0005538"]).score
    r0001 = score_candidate(_CANDIDATES["CAND_0000001"]).score
    assert r5538 > r0001, "plain-language genuine fit should outrank the keyword-stuffer trap"


def test_honeypot_is_credibility_flagged():
    if not _CANDIDATES:
        return
    result = score_candidate(_CANDIDATES["CAND_0003430"])
    assert result.credibility_flagged


def test_plain_language_flag_set_correctly():
    if not _CANDIDATES:
        return
    result = score_candidate(_CANDIDATES["CAND_0005538"])
    assert result.is_plain_language_profile


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
