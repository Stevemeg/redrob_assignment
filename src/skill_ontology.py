"""
Skill ontology: maps the dataset's 133 distinct skill tokens into semantic
clusters, so the matcher can recognize that a candidate who lists
"Vector Representations" and "Content Matching" is making the same claim as
a candidate who lists "Embeddings" and "RAG" -- just in plainer language.

This directly implements the JD's explicit instruction:
    "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
    profile, but if their career history shows they built a recommendation
    system at a product company, they're a fit."

Design note: clusters are built from manual inspection of the actual
candidate pool (see notebooks/explore_skills.py), not guessed from priors.
The dataset turned out to have three clean strata:
  - ~12k count "generic" skills (HTML, SAP, Accounting...) -> irrelevant to
    this JD, used only as negative signal (see is_offdomain_skill).
  - ~1.3k-5.2k count AI/ML/retrieval skills, split into a "popular/buzzwordy"
    band (~4.7k-5.2k: LangChain, RAG, Embeddings, CV/speech stack) and a
    "specialist infra" band (~1.3k-1.4k: Python, PyTorch, FAISS-adjacent
    vector DBs, Learning to Rank, BM25, NLP, Haystack, LlamaIndex).
  - A long tail of 1-7 count "plain language" tokens (Search Backend, Text
    Encoders, Vector Representations, ...) that never co-occur with the
    buzzword cluster -- this is the deliberately planted Tier-5 vocabulary.
"""

from __future__ import annotations


def _norm(s: str) -> str:
    return s.strip().lower()


# ---------------------------------------------------------------------------
# Semantic clusters. Each cluster groups tokens that represent the *same*
# underlying competency. Cluster membership is what powers adjacency: if a
# candidate lacks a specific JD-required token but has ANY other token in
# the same cluster, they get partial/full credit, not zero.
# ---------------------------------------------------------------------------

SKILL_CLUSTERS: dict[str, set[str]] = {
    "retrieval_ir": {
        "information retrieval", "information retrieval systems", "bm25",
        "search & discovery", "search backend", "search infrastructure",
        "indexing algorithms", "elasticsearch", "opensearch",
    },
    "vector_search": {
        "vector search", "semantic search", "embeddings", "sentence transformers",
        "vector representations", "text encoders", "faiss", "pinecone", "weaviate",
        "qdrant", "milvus", "pgvector",
    },
    "rag_llm_apps": {
        "rag", "langchain", "llamaindex", "haystack", "prompt engineering",
        "llms", "document processing", "content matching",
    },
    "ranking_recsys": {
        "recommendation systems", "ranking systems", "learning to rank",
        "content matching",
    },
    "llm_finetuning": {
        "fine-tuning llms", "lora", "qlora", "peft", "model adaptation",
        "hugging face transformers",
    },
    "ml_core": {
        "machine learning", "deep learning", "data science", "feature engineering",
        "statistical modeling", "scikit-learn", "tensorflow", "pytorch",
        "reinforcement learning",
    },
    "nlp": {
        "nlp", "natural language processing", "embeddings", "rag",
    },
    "computer_vision": {
        "computer vision", "image classification", "object detection", "cnn",
        "yolo", "opencv", "diffusion models", "gans",
    },
    "speech": {
        "speech recognition", "asr", "tts",
    },
    "mlops_infra": {
        "mlops", "mlflow", "weights & biases", "bentoml", "kubeflow",
        "workflow orchestration", "open-source ml libraries",
    },
    "time_series_forecasting": {
        "time series", "forecasting",
    },
    "core_programming": {
        "python",
    },
    "data_engineering": {
        "data pipelines", "etl", "spark", "airflow", "kafka", "hadoop",
        "apache beam", "apache flink", "dbt", "databricks", "bigquery",
        "snowflake",
    },
}

# Reverse index: skill token (normalized) -> set of cluster names it belongs to.
_SKILL_TO_CLUSTERS: dict[str, set[str]] = {}
for _cluster, _tokens in SKILL_CLUSTERS.items():
    for _t in _tokens:
        _SKILL_TO_CLUSTERS.setdefault(_norm(_t), set()).add(_cluster)


# Generic / off-domain skills: irrelevant-to-harmful signal for *this* JD.
# A profile dominated by these with no AI/retrieval cluster presence is
# almost certainly a non-fit, regardless of title.
GENERIC_OFFDOMAIN_SKILLS = {
    "html", "css", "redux", "angular", "vue.js", "figma", "salesforce crm",
    "sales", "accounting", "agile", "excel", "project management", "scrum",
    "illustrator", "photoshop", "tally", "six sigma", "seo", "sap",
    "content writing", "marketing", "powerpoint", "next.js", "node.js",
    "react", "typescript", "javascript", "java", "go", "rust", "graphql",
    "webpack", "tailwind", "spring boot", "django", "flask", "fastapi",
    "rest apis", "microservices", "mongodb", "postgresql", "redis", "sql",
    "gcp", "aws", "azure", "docker", "kubernetes", "terraform", "ci/cd",
    "grpc",
}

# The "plain-language" long-tail vocabulary identified by frequency analysis
# (count <= 10 in the 100k pool, never co-occurring with buzzword terms).
# Membership here doesn't change scoring directly (cluster membership above
# already covers them) -- it's used to flag "Tier 5 candidate" in reasoning
# text, since reviewers should be able to see *why* a low-keyword-count
# candidate ranked highly.
PLAIN_LANGUAGE_TIER5_TOKENS = {
    "information retrieval systems", "search backend", "text encoders",
    "vector representations", "content matching", "model adaptation",
    "ranking systems", "search & discovery", "workflow orchestration",
    "search infrastructure", "indexing algorithms", "open-source ml libraries",
    "natural language processing", "document processing",
}


def clusters_for_skill(skill_name: str) -> set[str]:
    """Return the semantic cluster(s) a raw skill string belongs to."""
    return _SKILL_TO_CLUSTERS.get(_norm(skill_name), set())


def candidate_cluster_coverage(skill_names: list[str]) -> dict[str, list[str]]:
    """
    Given a candidate's raw skill name list, return {cluster_name:
    [matching raw skill names]} for every cluster they have any presence in.
    """
    coverage: dict[str, list[str]] = {}
    for raw in skill_names:
        for cluster in clusters_for_skill(raw):
            coverage.setdefault(cluster, []).append(raw)
    return coverage


def is_offdomain_skill(skill_name: str) -> bool:
    return _norm(skill_name) in GENERIC_OFFDOMAIN_SKILLS


def is_plain_language_tier5_skill(skill_name: str) -> bool:
    return _norm(skill_name) in PLAIN_LANGUAGE_TIER5_TOKENS


def jd_group_coverage(skill_names: list[str], jd_group_tokens: set[str]) -> tuple[float, list[str]]:
    """
    Score how well a candidate's skills cover a JD requirement group, using
    cluster-based adjacency rather than exact string match.

    Returns (coverage_fraction, matched_raw_skill_names).

    coverage_fraction is the fraction of clusters touched by jd_group_tokens
    that the candidate has *any* presence in -- this is what lets a
    "Vector Representations" candidate get full credit against a JD
    requirement written in terms of "Embeddings" / "Pinecone".
    """
    jd_clusters: set[str] = set()
    for tok in jd_group_tokens:
        jd_clusters |= clusters_for_skill(tok)
    if not jd_clusters:
        return 0.0, []

    cand_clusters = candidate_cluster_coverage(skill_names)
    touched = jd_clusters & cand_clusters.keys()
    matched_raw: list[str] = []
    for c in touched:
        matched_raw.extend(cand_clusters[c])

    return len(touched) / len(jd_clusters), sorted(set(matched_raw))
