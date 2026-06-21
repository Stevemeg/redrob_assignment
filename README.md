# Redrob Challenge — Candidate Ranker

Ranks the 100,000-candidate pool against the "Senior AI Engineer — Founding
Team" job description and produces the top-100 submission CSV.

## Quick start

```bash
pip install -r requirements.txt   # only needed for eval/, ranking itself is stdlib-only
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Runtime on the full 100K-candidate pool: **~15-90 seconds** wall-clock,
CPU-only, no network calls, well inside the 5-minute / 16GB / no-GPU
compute budget in `submission_spec.md` Section 3. (Also works directly on
the gzipped `candidates.jsonl.gz` distributed in the official bundle — no
need to unzip first.)

## What this is

A rule-based, fully explainable ranking system over the candidate pool's
structured fields (profile, career history, skills, education, and the 23
`redrob_signals` behavioral fields) — deliberately **not** an embeddings/
LLM pipeline. See `ARCHITECTURE.md` for the full reasoning, but in short:
the compute constraints (no network, CPU-only, 5 min) make heavyweight ML
risky, and the data being structured JSON (not free-text resumes) means a
transparent feature-scoring approach is actually a better fit for the task
than semantic similarity would be, not just a compute-budget compromise.

## Repository layout

```
rank.py                    Entry point: streams candidates.jsonl, scores,
                            writes top-100 CSV.
src/
  jd_profile.py             Structured JD requirements (the single source
                             of truth every scoring component reads from).
  skill_ontology.py          Skill clustering for adjacency/transferable-
                             skill detection (e.g. "Vector Representations"
                             ~ "Embeddings").
  credibility.py             Honeypot / resume-credibility detection.
  title_classifier.py        Title/seniority scoring (primary defense
                             against the keyword-stuffer trap).
  behavioral_signals.py      23 redrob_signals -> one availability/
                             engagement multiplier.
  career_fit.py              Title-chaser, consulting-only, location/visa
                             checks.
  experience_education.py    Soft experience-band and education scoring.
  scorer.py                  Combines all of the above into the final
                             composite score.
  reasoning.py                Generates the CSV's `reasoning` column from
                             the same fields used for scoring (no
                             hallucination by construction).
eval/
  baseline_keyword.py         Baseline 1: pure keyword matching.
  baseline_tfidf.py           Baseline 2: TF-IDF cosine similarity.
  proxy_ground_truth.py       Transparent proxy relevance tiering for
                             offline evaluation (we don't have the real
                             hidden ground truth -- see ARCHITECTURE.md
                             Section 8 for the methodological caveat).
  run_eval.py                  NDCG@10/NDCG@50/MAP/P@10 comparison harness.
tests/
  test_*.py                    27 unit + end-to-end tests.
run_tests.py                  Runs the full test suite.
ARCHITECTURE.md              Design rationale, validated against the real
                             dataset, written for Stage 4/5 review.
submission_metadata.yaml      Portal metadata mirror.
```

## Running the tests

```bash
python run_tests.py
```

27 tests across credibility detection, skill ontology adjacency, title
classification, and end-to-end scorer behavior on four hand-validated
reference candidates (a genuine strong fit, a plain-language Tier-5
candidate, a keyword-stuffer trap, and a honeypot).

## Running the offline evaluation

```bash
python eval/run_eval.py --candidates ./candidates.jsonl --sample-size 20000
```

Compares our ranker against keyword-matching and TF-IDF baselines on the
same weighted-composite metric the real submission is scored on. See
`ARCHITECTURE.md` Section 8 for results and an honest discussion of the
proxy ground truth's limitations.

## Notes on the data bundle

While building this, we found the bundled `redrob_signals_doc.docx` is
missing the "trap candidates and signal envelopes" section the README
promises (it ends at signal #23 with no further content) — confirmed by
re-extracting via pandoc directly rather than relying on a single text
extraction tool. We relied on `submission_spec.md` Section 7 instead, which
does fully describe the honeypot mechanism. Flagging this per the README's
own "if you find a bug in the bundle" instruction.
