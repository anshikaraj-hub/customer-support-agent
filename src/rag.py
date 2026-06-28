# Task 6: RAG Pipeline
# Retrieves relevant information from company documents using TF-IDF similarity.
# No external API or model download required - runs fully offline.

import os
import math
import re
from typing import List
from collections import Counter

# Path to the docs folder
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

# The four knowledge base documents
KNOWLEDGE_BASE_FILES = {
    "company_policy":  "company_policy.txt",
    "pricing_guide":   "pricing_guide.txt",
    "technical_manual": "technical_manual.txt",
    "faq":             "faq.txt",
}

# Global in-memory index (built once on first import)
_index: List[dict] = []


def tokenize(text: str) -> List[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """Split a document into overlapping word chunks for better retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_knowledge_base():
    """Load documents, chunk them, and build a TF-IDF index."""
    global _index
    if _index:
        return  # Already built, skip

    print("[RAG] Building knowledge base from documents...")
    all_chunks = []

    for doc_name, filename in KNOWLEDGE_BASE_FILES.items():
        filepath = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[RAG] Warning: {filepath} not found, skipping.")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "id":     f"{doc_name}_chunk_{i}",
                "source": doc_name.replace("_", " ").title(),
                "text":   chunk,
                "tokens": Counter(tokenize(chunk)),
            })

    # Compute IDF scores across all chunks
    N = len(all_chunks)
    df = Counter()
    for chunk in all_chunks:
        for term in set(chunk["tokens"].keys()):
            df[term] += 1

    # Build TF-IDF vector for each chunk
    for chunk in all_chunks:
        total = sum(chunk["tokens"].values()) or 1
        chunk["tfidf"] = {
            term: (cnt / total) * (math.log((N + 1) / (df[term] + 1)) + 1)
            for term, cnt in chunk["tokens"].items()
        }

    _index = all_chunks
    print(f"[RAG] Indexed {len(_index)} chunks from {len(KNOWLEDGE_BASE_FILES)} documents.")


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    """Calculate cosine similarity between two TF-IDF vectors."""
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot = sum(vec1[t] * vec2[t] for t in common)
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


def retrieve_context(query: str, n_results: int = 3) -> str:
    """
    Find the most relevant document chunks for a given query.
    Returns a formatted string ready to be injected into an LLM prompt.
    """
    if not _index:
        build_knowledge_base()

    # Build TF-IDF vector for the query
    query_tokens = Counter(tokenize(query))
    total = sum(query_tokens.values()) or 1
    query_vec = {t: c / total for t, c in query_tokens.items()}

    # Score every chunk
    scored = sorted(
        [(cosine_similarity(query_vec, chunk["tfidf"]), chunk) for chunk in _index],
        key=lambda x: x[0],
        reverse=True
    )

    top = scored[:n_results]

    if not top or top[0][0] == 0:
        return "No relevant information found in the knowledge base."

    parts = []
    for score, chunk in top:
        relevance = round(score * 100, 1)
        parts.append(f"[Source: {chunk['source']} | Relevance: {relevance}%]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


# Build the index when this file is imported
build_knowledge_base()