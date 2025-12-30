#!/usr/bin/env python3
"""
Build a local vector index from PubMed JSONL.

Input:  data/pubmed.jsonl  (produced by scripts/pubmed_fetch.py)
Output: data/index.faiss   (FAISS index)
        data/meta.jsonl    (metadata aligned with FAISS rows)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

from tqdm import tqdm

# sentence-transformers
from sentence_transformers import SentenceTransformer

# FAISS (CPU)
import faiss
import numpy as np


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ensure_outdir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _build_text(rec: Dict[str, Any]) -> str:
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    journal = (rec.get("journal") or "").strip()
    year = (rec.get("pub_year") or "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if journal or year:
        parts.append(f"Source: {journal} ({year})".strip())
    return "\n".join(parts).strip()


def main():
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
        raise FileNotFoundError(f"Input not found: {args.inp}. Run scripts/pubmed_fetch.py first.")

    _ensure_outdir(args.index_out)
    _ensure_outdir(args.meta_out)
