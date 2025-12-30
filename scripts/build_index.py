#!/usr/bin/env python3
"""
Build a local FAISS vector index from PubMed JSONL.

Input:
  data/pubmed.jsonl   (produced by scripts/pubmed_fetch.py)

Output:
  data/index.faiss    (FAISS index, cosine similarity)
  data/meta.jsonl     (metadata aligned with FAISS rows)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_outdir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def build_text(rec: Dict[str, Any]) -> str:
    """
    Build embedding text from a PubMed record.
    Robust to missing fields.
    """
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    journal = (rec.get("journal") or "").strip()

    year_val = rec.get("year") or rec.get("pub_year")
    year = str(year_val).strip() if year_val is not None else ""

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    src_bits: List[str] = []
    if journal:
        src_bits.append(journal)
    if year:
        src_bits.append(year)
    if src_bits:
        parts.append("Source: " + " | ".join(src_bits))

    return "\n".join(parts).strip()


# --------------------------------------------------
# Main
# --------------------------------------------------

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
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise FileNotFoundError(
            f"Input not found: {args.inp}. Run scripts/pubmed_fetch.py first."
        )

    ensure_outdir(args.index_out)
    ensure_outdir(args.meta_out)

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    # ------------------------------
    # Load & prepare documents
    # ------------------------------
    for i, rec in enumerate(iter_jsonl(args.inp), start=1):
        text = build_text(rec)
        if not text:
            continue

        texts.append(text)
        meta.append(
            {
                "pmid": rec.get("pmid"),
                "title": rec.get("title"),
                "abstract": rec.get("abstract"),
                "year": rec.get("year") or rec.get("pub_year"),
                "journal": rec.get("journal"),
                "url": rec.get("url")
                or (f"https://pubmed.ncbi.nlm.nih.gov/{rec.get('pmid')}/"
                    if rec.get("pmid") else None),
                "query_source": rec.get("query_source"),
            }
        )

        if args.max_docs and i >= args.max_docs:
            break

    if not texts:
        raise SystemExit("No valid documents found in input JSONL.")

    print(f"[build_index] Documents loaded: {len(texts)}")

    # ------------------------------
    # Embedding
    # ------------------------------
    print(f"[build_index] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[start : start + args.batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine-ready
        )
        embeddings.append(emb.astype("float32"))

    X = np.vstack(embeddings)
    dim = X.shape[1]

    print(f"[build_index] Embedding matrix shape: {X.shape}")

    # ------------------------------
    # FAISS index (cosine similarity)
    # ------------------------------
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index/meta mismatch: index.ntotal={index.ntotal}, meta={len(meta)}"
        )

    faiss.write_index(index, args.index_out)
    print(f"[build_index] Saved FAISS index: {args.index_out}")

    # ------------------------------
    # Save metadata (aligned!)
    # ------------------------------
    with open(args.meta_out, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[build_index] Saved metadata: {args.meta_out}")
    print("[build_index] Done.")


if __name__ == "__main__":
    main()
