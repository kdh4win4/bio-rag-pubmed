#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_meta(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def embed(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype("float32")


def retrieve(
    question: str,
    index_path: str,
    meta_path: str,
    top_k: int,
    model_name: str,
) -> List[Tuple[float, Dict[str, Any]]]:

    index = faiss.read_index(index_path)
    meta = load_meta(meta_path)
    model = SentenceTransformer(model_name)

    qvec = embed(model, [question])
    scores, idxs = index.search(qvec, top_k)

    out = []
    for s, i in zip(scores[0], idxs[0]):
        if 0 <= i < len(meta):
            out.append((float(s), meta[i]))
    return out


def call_openai(system: str, user: str, model: str, temperature: float) -> str:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--index_path", default="data/index.faiss")
    ap.add_argument("--meta_path", default="data/meta.jsonl")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", choices=["none", "openai"], default="none")
    ap.add_argument("--llm_model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    hits = retrieve(
        question=args.question,
        index_path=args.index_path,
        meta_path=args.meta_path,
        top_k=args.top_k,
        model_name=args.embed_model,
    )

    print("\n=== Top-K Retrieved Papers ===")
    for i, (score, m) in enumerate(hits, 1):
        print(f"\n[{i}] score={score:.4f}")
        print("Title:", m.get("title", ""))
        print("PMID:", m.get("pmid", ""))
        print("Journal:", m.get("journal", ""), m.get("pub_year", ""))
        print("URL:", m.get("url", ""))
        abs_txt = (m.get("abstract") or "").strip()
        if abs_txt:
            print("Evidence:", abs_txt[:400] + ("..." if len(abs_txt) > 400 else ""))
        else:
            print("Evidence: (no abstract)")

    if args.llm == "openai":
        context = []
        for i, (_, m) in enumerate(hits, 1):
            block = f"[{i}] {m.get('title','')}\n{m.get('abstract','(no abstract)')}"
            context.append(block)

        system = (
            "You are a biomedical assistant. "
            "Answer ONLY using the evidence provided. "
            "If evidence is insufficient, say so explicitly. "
            "Cite using [1], [2], etc."
        )

        user = (
            f"Question:\n{args.question}\n\n"
            f"Evidence:\n" + "\n\n".join(context)
        )

        answer = call_openai(
            system=system,
            user=user,
            model=args.llm_model,
            temperature=args.temperature,
        )

        print("\n=== LLM Answer (Grounded) ===\n")
        print(answer)


if __name__ == "__main__":
    main()
