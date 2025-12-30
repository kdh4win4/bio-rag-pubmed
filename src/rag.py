#!/usr/bin/env python3
"""
RAG over local PubMed mini-corpus (FAISS + SentenceTransformers).

Prereqs:
  - data/index.faiss
  - data/meta.jsonl
Built by:
  python scripts/pubmed_fetch.py --query "..." --max_results 50 --out data/pubmed.jsonl
  python scripts/build_index.py --in data/pubmed.jsonl --index_out data/index.faiss --meta_out data/meta.jsonl

Run:
  python src/rag.py --question "..." --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_meta(meta_path: str | Path) -> List[Dict[str, Any]]:
    meta_path = str(meta_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    rows: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize(vecs: np.ndarray) -> np.ndarray:
    # L2-normalize for cosine via inner product
    faiss.normalize_L2(vecs)
    return vecs


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # already normalized
    )
    # ensure float32 for faiss
    return np.asarray(embs, dtype="float32")


def build_query_text(question: str) -> str:
    q = (question or "").strip()
    if not q:
        raise ValueError("Empty question.")
    return q


def retrieve(
    question: str,
    index_path: str | Path = "data/index.faiss",
    meta_path: str | Path = "data/meta.jsonl",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    index_path = str(index_path)
    meta_path = str(meta_path)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path} (run scripts/build_index.py)")
    meta = load_meta(meta_path)

    index = faiss.read_index(index_path)

    model = SentenceTransformer(model_name)

    qtext = build_query_text(question)
    q_emb = embed_texts(model, [qtext])  # (1, dim)

    # Search
    scores, idx = index.search(q_emb, top_k)  # scores shape (1, top_k)
    scores = scores[0].tolist()
    idx = idx[0].tolist()

    results: List[Tuple[float, Dict[str, Any]]] = []
    for s, i in zip(scores, idx):
        if i < 0 or i >= len(meta):
            continue
        results.append((float(s), meta[i]))
    return results


def synthesize_answer(question: str, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
    """
    Offline 'summary' without calling an LLM:
    - Pull 1-2 key sentences from each abstract (very simple heuristic).
    - Return a cautious draft.
    """
    bullet_points: List[str] = []
    for _, m in hits:
        abs_txt = (m.get("abstract") or "").strip()
        if not abs_txt:
            continue
        # naive sentence split
        sents = [s.strip() for s in abs_txt.replace("\n", " ").split(".") if s.strip()]
        pick = sents[:2]  # first 1-2 sentences
        if pick:
            bullet_points.append("- " + ". ".join(pick) + ".")
    if not bullet_points:
        return (
            "I couldn't extract usable abstract sentences from the retrieved records. "
            "Try increasing --max_results when fetching, or use a query that returns papers with abstracts."
        )

    draft = [
        "Draft answer (offline, evidence-snippet based):",
        f"Question: {question}",
        "",
        "Key points suggested by retrieved abstracts:",
        *bullet_points[:10],
        "",
        "Note: This is a lightweight, non-LLM synthesis. If you want an LLM-style narrative answer, we can add an optional OpenAI/API key flow later.",
    ]
    return "\n".join(draft)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--top_k", type=int, default=5, help="How many papers to retrieve")
    ap.add_argument("--index_path", default="data/index.faiss")
    ap.add_argument("--meta_path", default="data/meta.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    hits = retrieve(
        question=args.question,
        index_path=args.index_path,
        meta_path=args.meta_path,
        model_name=args.model,
        top_k=args.top_k,
    )

    print("\n=== Top-K Retrieved Papers ===")
    if not hits:
        print("(no hits)")
        return

    for rank, (score, m) in enumerate(hits, start=1):
        pmid = m.get("pmid", "")
        title = (m.get("title") or "").strip()
        journal = (m.get("journal") or "").strip()
        year = (m.get("pub_year") or "").strip()
        url = m.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")
        print(f"\n[{rank}] score={score:.4f}")
        print(f"PMID: {pmid}")
        print(f"Title: {title}")
        print(f"Source: {journal} ({year})")
        if url:
            print(f"URL: {url}")

        # show a short evidence snippet
        abs_txt = (m.get("abstract") or "").strip().replace("\n", " ")
        if abs_txt:
            snippet = abs_txt[:400] + ("..." if len(abs_txt) > 400 else "")
            print(f"Evidence snippet: {snippet}")
        else:
            print("Evidence snippet: (no abstract)")

    print("\n=== Answer ===")
    print(synthesize_answer(args.question, hits))
    print()


if __name__ == "__main__":
    main()
