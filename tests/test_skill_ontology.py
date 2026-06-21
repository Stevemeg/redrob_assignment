"""
Unit tests for skill_ontology.py -- specifically the cluster-adjacency
behavior that lets plain-language skill descriptions get credit against
buzzword-phrased JD requirements.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skill_ontology import (
    jd_group_coverage, clusters_for_skill, is_offdomain_skill,
    is_plain_language_tier5_skill,
)
import jd_profile as jd


def test_plain_language_skills_match_buzzword_jd_requirement():
    # "Vector Representations" is the plain-language equivalent of
    # "Embeddings" -- both belong to the vector_search cluster.
    plain_skills = ["Vector Representations", "Content Matching"]
    cov, matched = jd_group_coverage(plain_skills, jd.MUST_HAVE_SKILL_GROUPS["embeddings_retrieval"])
    assert cov > 0, "plain-language skills should match cluster-adjacent JD requirements"
    assert "Vector Representations" in matched


def test_offdomain_skills_score_zero():
    hr_skills = ["Excel", "PowerPoint", "Sales", "Six Sigma"]
    for group, tokens in jd.MUST_HAVE_SKILL_GROUPS.items():
        cov, matched = jd_group_coverage(hr_skills, tokens)
        assert cov == 0.0, f"non-technical skills should not match {group}"
        assert matched == []


def test_known_generic_skills_flagged_offdomain():
    assert is_offdomain_skill("HTML")
    assert is_offdomain_skill("salesforce crm")
    assert not is_offdomain_skill("Embeddings")


def test_known_tier5_tokens_flagged():
    assert is_plain_language_tier5_skill("Search Backend")
    assert is_plain_language_tier5_skill("text encoders")
    assert not is_plain_language_tier5_skill("Python")


def test_exact_buzzword_match_also_works():
    # Sanity: the cluster system shouldn't break the trivial exact-match case.
    skills = ["Pinecone", "RAG", "Embeddings"]
    cov, matched = jd_group_coverage(skills, jd.MUST_HAVE_SKILL_GROUPS["embeddings_retrieval"])
    assert cov > 0
    assert "Pinecone" in matched


def test_cluster_membership_is_consistent():
    # Embeddings and Vector Representations should land in the same cluster.
    c1 = clusters_for_skill("Embeddings")
    c2 = clusters_for_skill("Vector Representations")
    assert c1 & c2, "Embeddings and Vector Representations should share a cluster"


def test_unknown_skill_returns_empty_cluster():
    assert clusters_for_skill("CompletelyMadeUpSkillXYZ") == set()


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
