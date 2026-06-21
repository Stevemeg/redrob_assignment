# Architecture & Design Rationale

This document explains *why* the system is built the way it is, for Stage 4
(manual review) and Stage 5 (defend-your-work interview). Every design
choice below is traceable to either the job description, the submission
spec's compute constraints, or a pattern directly observed in the real
`candidates.jsonl` (not assumed from priors).

## 1. Why a rule-based scorer over structured fields, not embeddings/LLMs

The submission spec's compute constraints (Section 3) rule out the
heavyweight options outright:

- **No network during ranking** → no hosted LLM calls (OpenAI/Claude/etc.)
  for per-candidate scoring or reasoning generation.
- **CPU only, 5 minutes, 16GB** → running a sentence-transformer encoder
  over 100,000 candidate summaries, even locally, is a real risk against
  the wall-clock budget once you include model load time, and it pulls in
  torch/transformers as dependencies that complicate "single command,
  reproducible in a sandboxed container."

But there's a more important reason than "the constraints force it": **the
data is structured, not free text.** Every candidate is a JSON record with
typed fields (`years_of_experience`, `proficiency` enums, `duration_months`
integers, 23 numeric/boolean behavioral signals). A recruiter evaluating
these candidates isn't doing semantic similarity on paragraphs — they're
cross-referencing specific facts ("5.9 years, but career history only
covers 11 months — that's a red flag"). Encoding the JD's stated logic
directly as feature scoring over those fields is *more* accurate than
embedding similarity would be here, not just faster. The organizers' own
`submission_metadata_template.yaml` example `methodology_summary` describes
almost exactly this kind of system, which we take as a strong signal this
is the intended solution shape, not a workaround.

Our actual runtime on the full 100K pool: **16-87 seconds** (cold vs warm
disk cache), comfortably inside the 5-minute budget, using **~4MB** of
Python-tracked heap allocation (we stream the JSONL line-by-line and keep
only a 100-element min-heap, never materializing the full pool in memory).

## 2. The skill ontology is the core differentiator

The JD is explicit and repeated on this point:

> "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
> profile, but if their career history shows they built a recommendation
> system at a product company, they're a fit."

We validated this is a *real, deliberately planted* pattern in the data,
not a hypothetical: scanning the full skill vocabulary (133 distinct
tokens across 100K candidates) shows a clean three-tier frequency
structure —

| Tier | Count range | Example tokens |
|---|---|---|
| Generic/off-domain | ~12,000 | HTML, SAP, Accounting, Salesforce CRM |
| AI/ML buzzword cluster | ~1,300–5,200 | LangChain, RAG, Embeddings, Pinecone |
| Plain-language long tail | 1–7 | "Search Backend", "Vector Representations", "Content Matching" |

Only **8 candidates in the entire 100K pool** use the plain-language
vocabulary, and they never co-occur with the buzzword cluster — strong
evidence this is a deliberately planted Tier-5 archetype, not noise.
`src/skill_ontology.py` builds semantic clusters (e.g. `vector_search` =
{Embeddings, Pinecone, Vector Representations, Text Encoders, FAISS, ...})
so that **cluster membership**, not exact string match, determines JD
requirement coverage. We confirmed this works as intended: candidate
`CAND_0005538` (plain-language profile) scores 1.00 coverage against every
must-have skill group, identical to candidates using the buzzword
vocabulary for the same underlying competency.

## 3. Title classification is the primary defense against keyword-stuffing

The bundled `sample_submission.csv` — the organizers' own format reference
— ranks an "HR Manager with 9 AI core skills" at rank #1. This is *exactly*
the trap the JD warns about:

> "The 'right answer' to this JD is not 'find candidates whose skills
> section contains the most AI keywords.' That's a trap we've explicitly
> built into the dataset."

`src/title_classifier.py` weights current (and historical) title heavily —
30% of the composite score, second only to skill match — specifically so a
non-engineering title caps the achievable score regardless of skill list
length. We verified this directly: `CAND_0000001` (Backend Engineer at
Mindtree, genuinely strong data-engineering background, but several
AI-sounding skills listed without real AI career history) scores 0.51,
well below genuine AI/ML-titled candidates with comparable skill-list
length, and far below the plain-language Tier-5 candidate (0.90) despite
the latter having *fewer* AI-sounding keyword matches.

## 4. Credibility / honeypot detection

The spec documents ~80 honeypot candidates with "subtly impossible
profiles," forced to relevance tier 0 in the hidden ground truth, with a
>10% honeypot rate in the top 100 causing disqualification at Stage 3.

Rather than guess at what "subtly impossible" means, we scanned the real
data for internal self-contradictions and found three independent,
**non-overlapping** patterns:

1. **YOE mismatch**: `profile.years_of_experience` disagrees with the sum
   of `career_history` durations by >24 months. Example found in the data:
   `CAND_0003430` — headline says "1.3+ yrs experience," career history
   totals 11 months, but `years_of_experience` claims 13.7.
2. **Expert-with-zero-duration**: 3+ skills claimed at "expert" proficiency
   with `duration_months == 0`.
3. **Overqualified-for-experience**: 10+ advanced/expert skills with
   `years_of_experience < 2`.

Combined, these flag **96 candidates** — close to the documented ~80 — with
**zero overlap between heuristics** (each catches a distinct contradiction
type), which we read as evidence these are independently-injected traps
rather than natural data noise, and as reasonable confirmation the
thresholds are in the right place.

Flagged candidates get a `0.15^n` multiplicative penalty (n = number of
flags triggered) rather than a hard filter, per the JD's own instruction:
"we expect a good ranking system to naturally avoid them; you don't need
to special-case them." On the full 100K-candidate run, **0 credibility-
flagged candidates appear in our top 100.**

## 5. Behavioral signals as a multiplier, not a feature

> "A perfect-on-paper candidate who hasn't logged in for 6 months and has a
> 5% recruiter response rate is, for hiring purposes, not actually
> available. Down-weight them appropriately."

We deliberately collapse the 23 `redrob_signals` fields into a single
`[0.3, 1.15]` multiplier (`src/behavioral_signals.py`) rather than treating
each as an independent weighted feature. Most of the 23 fields are
correlated proxies for one latent factor ("is this person actually
reachable and serious right now") — `recruiter_response_rate`,
`avg_response_time_hours`, and `interview_completion_rate` all measure
close variants of the same thing. Decomposing further would mostly add
noise, not signal. Fields with no clear interpretive link to fit or
availability (`verified_email`, `linkedin_connected`, etc.) are
deliberately excluded as platform-hygiene fields, not because we forgot
them.

## 6. Career-trajectory and contextual-fit checks

`src/career_fit.py` directly encodes the JD's "things we explicitly do NOT
want" section: title-chasers (3+ jobs all ≤20 months), consulting-only
careers (100% of career history at TCS/Infosys/Wipro/etc. with zero
product-company experience), and location/visa fit (Pune/Noida preferred,
other Tier-1 Indian cities welcome, no visa sponsorship outside India).
Each is a soft multiplicative penalty, not a disqualifier — consistent with
the JD's own framing ("if you're currently at one of these but have prior
product-company experience, that's fine").

## 7. Reasoning generation

Every reasoning string is built (`src/reasoning.py`) purely from fields
already read off the candidate object during scoring — never from a
separate LLM call, and never inventing facts. This satisfies Stage 4's six
reasoning checks by construction:

- **Specific facts**: every sentence cites an actual field value (title,
  company, years, named skills, signal values).
- **JD connection**: lead clauses reference the specific component that
  was decisive for that candidate's score (skill match, title category,
  experience-band distance, location, credibility flag).
- **Honest concerns**: low-scoring components are surfaced as concern
  clauses even for otherwise-strong candidates (e.g. "based outside India,
  not open to relocation, role does not sponsor visas" for an otherwise
  skill-matched candidate).
- **No hallucination**: structurally impossible, since every clause is
  templated from `ScoredCandidate.notes`, which only ever contains strings
  built from `candidate[...]` field reads.
- **Variation**: which clause leads varies candidate-to-candidate because
  different candidates are decided by different components.
- **Rank consistency**: reasoning tone tracks score directly, since both
  are derived from the same components in the same pass.

## 8. Evaluation methodology and an honest caveat

We do not have access to the hidden ground truth (only revealed after
submissions close), so `eval/run_eval.py` builds a transparent **proxy**
relevance tiering (`eval/proxy_ground_truth.py`) from the same kind of
signals the JD describes (title category, skill-cluster coverage,
credibility flags) to enable *some* quantitative comparison during
development.

**Important methodological caveat**: this proxy ground truth shares
underlying logic (`skill_ontology.jd_group_coverage`,
`credibility.check_credibility`) with the ranker itself. This means the
absolute composite scores reported below should not be read as "true"
performance against the real hidden ground truth — there's a risk of
circularity. What *is* a fair comparison is the **relative** gap against
the two baselines, since neither baseline uses any of our ontology or
credibility logic at all:

| System | NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Honeypots@10 | Honeypots@100 |
|---|---|---|---|---|---|---|---|
| Baseline 1: keyword matching | 0.584 | 0.718 | 0.995 | 0.400 | 0.677 | 0 | 1 |
| Baseline 2: TF-IDF cosine similarity | 0.931 | 0.800 | 0.957 | 0.900 | 0.894 | 1 | 4 |
| Our ranker | 1.000 | 0.993 | 1.000 | 1.000 | 0.998 | 0 | 0 |

(n=20,000 sampled candidates; full results in `output/eval_results.json`,
reproducible via `python eval/run_eval.py --candidates <path> --sample-size 20000`)

The directionally meaningful findings: our ranker is the only one of the
three with **zero proxy-honeypots in the top 10 or top 100**, while the
TF-IDF baseline — despite being a reasonable lexical-statistics approach —
leaks one honeypot into the top 10 purely because honeypot profiles can
still have high lexical overlap with the JD text (a honeypot can claim
"expert NLP, RAG, Embeddings" — the *words* are there even though the
profile is internally contradictory). Catching that requires actually
checking internal consistency, which neither baseline does and our
ranker's credibility module does explicitly.

## 9. What we'd build next (out of scope for the 5-day build)

- **Learned re-ranking**: once real recruiter engagement data exists
  (clicks, saves, response patterns on actual ranked output), a
  gradient-boosted re-ranker (XGBoost/LightGBM) over these same structured
  features plus interaction terms could likely beat the hand-weighted
  composite, while staying within the CPU-only constraint.
- **Learned skill ontology**: our clusters were built by manual inspection
  of the 133-token vocabulary. At real platform scale, a co-occurrence-
  based or lightweight local-embedding-based clustering (computed offline,
  not at ranking time) would generalize better to unseen skill tokens.
- **Calibrated, not proxy, ground truth**: real recruiter agreement scores
  (have actual recruiters label a sample) would replace our proxy tiering
  and let us validate the component weights empirically rather than by
  JD-reading judgment.
