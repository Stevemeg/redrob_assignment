"""
Baseline 2: TF-IDF cosine similarity between the JD text and each
candidate's free-text fields (summary + career_history descriptions +
headline), using scikit-learn's TfidfVectorizer.

This is a step up from pure keyword counting -- it captures term frequency
and inverse document frequency weighting -- but it's still bag-of-words: it
has no concept of skill adjacency (LangChain vs LlamaIndex are unrelated
tokens to TF-IDF) and no understanding of structured fields like title,
experience band, or behavioral signals. It's included as Baseline 2 to show
the incremental value of moving from keywords to lexical-statistical
matching, before moving to the full structured + ontology-aware system.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JD_TEXT = """
Senior AI Engineer Founding Team. Own the intelligence layer: ranking,
retrieval, and matching systems. Production experience with embeddings
based retrieval systems sentence-transformers OpenAI embeddings BGE E5.
Production experience with vector databases or hybrid search
infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch
FAISS. Strong Python. Evaluation frameworks for ranking systems NDCG MRR
MAP offline online correlation A/B test. LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank models XGBoost neural. Recommendation systems search
retrieval ranking at scale to real users.
"""


def candidate_text(candidate: dict) -> str:
    parts = [candidate.get("profile", {}).get("headline", "")]
    parts.append(candidate.get("profile", {}).get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
    parts.extend(s["name"] for s in candidate.get("skills", []))
    return " ".join(parts)


def tfidf_rank(candidates: list[dict]) -> list[float]:
    """Returns cosine similarity of each candidate's text against the JD."""
    docs = [JD_TEXT] + [candidate_text(c) for c in candidates]
    vec = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf = vec.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sims.tolist()
