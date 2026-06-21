"""
Structured representation of the Redrob job description (Senior AI Engineer,
Founding Team). This module is the single source of truth for "what the JD
wants" -- every scoring component reads from here rather than hardcoding
JD facts inline. Keeping this separate also means re-targeting the ranker
at a different JD is a config change, not a code change.

Source: job_description.docx (bundled with the hackathon kit).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core experience band
# ---------------------------------------------------------------------------

EXPERIENCE_BAND = (5, 9)          # "5-9 years", explicitly a soft band, not a hard cutoff
EXPERIENCE_SOFT_MARGIN = 2.5      # years outside the band before the penalty gets severe

# ---------------------------------------------------------------------------
# Skills the JD calls "things you absolutely need" -- these are weighted
# heaviest in the skill-match component.
# ---------------------------------------------------------------------------

MUST_HAVE_SKILL_GROUPS = {
    # group_name: representative skill tokens in the dataset vocabulary that
    # count as evidence of this requirement. Membership in ANY group below
    # contributes to that group's coverage -- we don't need every token.
    "embeddings_retrieval": {
        "embeddings", "sentence transformers", "vector search", "semantic search",
        "rag", "information retrieval", "faiss", "bm25", "learning to rank",
        "vector representations", "text encoders", "indexing algorithms",
        "search infrastructure", "search backend", "ranking systems",
    },
    "vector_db_hybrid_search": {
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "faiss", "pgvector", "search & discovery", "search infrastructure",
    },
    "python_engineering": {
        "python",
    },
    "eval_frameworks": {
        "learning to rank", "bm25", "mlflow", "weights & biases", "feature engineering",
        "statistical modeling",
    },
}

# "Would like but won't reject for"
NICE_TO_HAVE_SKILL_GROUPS = {
    "llm_finetuning": {"lora", "qlora", "peft", "fine-tuning llms", "model adaptation"},
    "learning_to_rank_models": {"learning to rank", "xgboost"},
    "distributed_systems": {"kafka", "kubernetes", "kubeflow", "apache flink", "apache beam"},
}

# Skills that signal "framework tourist" rather than systems thinker when they
# dominate a profile with no infra/production skills alongside them.
FRAMEWORK_ENTHUSIAST_SIGNALS = {"langchain", "prompt engineering"}

# ---------------------------------------------------------------------------
# Disqualifying / heavily-penalized patterns, taken near-verbatim from the
# JD's "Things we explicitly do NOT want" and disqualifier sections.
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "hcltech",
}

PURE_VISION_SPEECH_ROBOTICS_TITLES = {
    "computer vision engineer",
}

# Industries that, combined with an otherwise-generic title, suggest pure
# non-product/consulting background.
SERVICES_INDUSTRY_MARKERS = {"it services", "consulting", "bpo"}

# ---------------------------------------------------------------------------
# Location preferences
# ---------------------------------------------------------------------------

PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram", "bengaluru", "bangalore"}
# "Outside India: case-by-case, no visa sponsorship" -> heavy penalty, not disqualification
SPONSORS_VISAS = False

# ---------------------------------------------------------------------------
# Notice period
# ---------------------------------------------------------------------------

IDEAL_NOTICE_DAYS = 30   # "we'd love sub-30-day notice... can buy out up to 30 days"
NOTICE_GRACE_DAYS = 30   # 30+ is "still in scope but the bar gets higher"

# ---------------------------------------------------------------------------
# Title archetypes used by the title/seniority component.
# ---------------------------------------------------------------------------

# Titles that are direct, on-paper matches for an AI/ML systems engineer.
CORE_AI_TITLES = {
    "ai engineer", "senior ai engineer", "lead ai engineer", "machine learning engineer",
    "senior machine learning engineer", "staff machine learning engineer",
    "applied ml engineer", "ml engineer", "junior ml engineer", "ai research engineer",
    "data scientist", "senior data scientist", "nlp engineer", "senior nlp engineer",
    "senior software engineer (ml)", "recommendation systems engineer", "search engineer",
    "ai specialist", "senior applied scientist",
}

# Titles that are adjacent / plausible Tier-5 backgrounds: real software and
# data engineering roles that could plausibly have built the underlying
# systems the JD wants, even without an "AI" label on the title.
ADJACENT_ENGINEERING_TITLES = {
    "software engineer", "senior software engineer", "backend engineer",
    "data engineer", "senior data engineer", "analytics engineer",
    "full stack developer", "devops engineer",
}

# Titles that are essentially never going to be a fit for this systems/ML
# engineering role, regardless of how many "AI" skill keywords are attached.
NON_ENGINEERING_TITLES = {
    "business analyst", "hr manager", "mechanical engineer", "accountant",
    "project manager", "customer support", "operations manager", "content writer",
    "sales executive", "civil engineer", "graphic designer", "marketing manager",
}

# Titles that are technical but in a different discipline than the JD wants
# (JD: "computer vision, speech, or robotics without significant NLP/IR exposure")
OFF_DOMAIN_TECH_TITLES = {
    "mobile developer", "frontend engineer", "java developer", ".net developer",
    "qa engineer", "cloud engineer",
}
