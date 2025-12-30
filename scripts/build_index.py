#!/usr/bin/env python3
"""
Build a local vector index from PubMed JSONL.

Input:  data/pubmed.jsonl  (produced by scripts/pubmed_fetch.py)
Output: data/index.faiss   (FAISS index)
        data/meta.jsonl    (metadata aligned with FAISS rows)

Design:
- Uses SentenceTransformers to embed each document (title+abstract+source)
- Builds a FAISS index for fast TopK retrieval
- Writes meta.jsonl so row i corresponds to FAISS vector i
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_outdir(filepath: str) -> None:
    out_dir = os.path.dirname(filepath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_text(rec: Dict[str, Any]) -> str:
    title = as_str(rec.get("title"))
    abstract = as_str(rec.get("abstract"))
    journal = as_str(rec.get("journal"))
    year = as_str(rec.get("pub_year"))

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if journal or year:
        src = "Source: "
        if journal and year:
            src += f"{journal} ({year})"
        elif journal:
            src += journal
        else:
            src += f"({year})"
        parts.append(src)

    return "\n".join(parts).strip()


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # mat: (n, d)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pubmed.jsonl", help="Input JSONL path")
    ap.add_argument("--index_out", default="data/index.faiss", help="FAISS index output path")
    ap.add_argument("--meta_out", default="data/meta.jsonl", help="Metadata JSONL output path")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--normalize", action="store_true", help="Use cosine similarity (IP on normalized vectors)")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise FileNotFoundError(f"Input not found: {args.inp}. Run scripts/pubmed_fetch.py first.")

    ensure_outdir(args.index_out)
    ensure_outdir(args.meta_out)

    # 1) Load + prepare texts & metas
    metas: List[Dict[str, Any]] = []
    texts: List[str] = []

    n = 0
    for rec in iter_jsonl(args.inp):
        text = build_text(rec)
        if not text:
            continue

        # Keep meta minimal + stable
        meta = {
            "pmid": as_str(rec.get("pmid")),
            "title": as_str(rec.get("title")),
            "journal": as_str(rec.get("journal")),
            "pub_year": as_str(rec.get("pub_year")),
            "doi": as_str(rec.get("doi")),
            "url": as_str(rec.get("url")),
            "query_source": as_str(rec.get("query_source")),
        }
        metas.append(meta)
        texts.append(text)
        n += 1
        if args.max_docs and n >= args.max_docs:
            break

    if not texts:
        raise RuntimeError("No valid documents found in JSONL. Check data/pubmed.jsonl contents.")

    # 2) Embed
    print(f"[build_index] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    embeddings: List[np.ndarray] = []
    bs = max(1, int(args.batch_size))

    for i in tqdm(range(0, len(texts), bs), desc="[build_index] Embedding"):
        batch = texts[i : i + bs]
        em = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        em = em.astype("float32")
        embeddings.append(em)

    X = np.vstack(embeddings).astype("float32")  # (N, D)

    # 3) Build FAISS index
    dim = X.shape[1]
    if args.normalize:
        X = l2_normalize(X)
        index = faiss.IndexFlatIP(dim)  # cosine via inner product on normalized vectors
    else:
        index = faiss.IndexFlatL2(dim)  # Euclidean

    index.add(X)

    # 4) Save
    faiss.write_index(index, args.index_out)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"[build_index] Done.")
    print(f"  docs:      {len(metas)}")
    print(f"  dim:       {dim}")
    print(f"  index_out: {args.index_out}")
    print(f"  meta_out:  {args.meta_out}")
    print(f"  metric:    {'cosine (IP+normalize)' if args.normalize else 'L2'}")


if __name__ == "__main__":
    main()
