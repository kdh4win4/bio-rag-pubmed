#!/usr/bin/env python3
"""
BioRAG-PubMed: end-to-end Q&A runner
- Loads FAISS index + metadata
- Retrieves TopK PubMed records
- Optionally calls local Ollama to draft an answer grounded in evidence
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import List, Dict, Any, Optional

from src.rag import BioRAG


def _format_hits(hits: List[Dict[str, Any]], max_abs_chars: int = 700) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        pmid = h.get("pmid", "NA")
        title = (h.get("title") or "").strip()
        year = h.get("year", "")
        score = h.get("score", None)
        abstract = (h.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > max_abs_chars:
            abstract = abstract[:max_abs_chars].rstrip() + "…"

        header = f"[{i}] PMID:{pmid}  Year:{year}  Score:{score:.4f}" if isinstance(score, float) else f"[{i}] PMID:{pmid}  Year:{year}"
        lines.append(header)
        if title:
            lines.append(f"Title: {title}")
        if abstract:
            lines.append(f"Abstract: {abstract}")
        lines.append("")  # blank line
    return "\n".join(lines).strip()


def _ollama_generate(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    # No extra dependency: uses requests (already in requirements.txt)
    import requests

    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.faiss", help="Path to FAISS index file")
    ap.add_argument("--meta", default="data/meta.jsonl", help="Path to metadata JSONL (aligned with index)")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--topk", type=int, default=5, help="How many papers to retrieve")
    ap.add_argument("--llm", choices=["none", "ollama"], default="none", help="LLM mode")
    ap.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "llama3.1"), help="Ollama model name")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), help="Ollama host URL")
    args = ap.parse_args()

    rag = BioRAG(index_path=args.index, meta_path=args.meta)
    rag.load()

    hits = rag.retrieve(args.question, topk=args.topk)

    print("\n=== TopK Evidence (PubMed) ===\n")
    print(_format_hits(hits))

    if args.llm == "none":
        print("\n=== Answer (evidence-only mode) ===\n")
        print(
            "LLM을 사용하지 않고 TopK 근거만 출력했습니다.\n"
            "근거 기반 요약 답변까지 자동으로 만들고 싶으면 `--llm ollama`로 실행하세요."
        )
        return

    # Build grounded prompt for Ollama
    evidence_blocks = []
    for i, h in enumerate(hits, start=1):
        pmid = h.get("pmid", "NA")
        title = (h.get("title") or "").strip()
        abstract = (h.get("abstract") or "").strip()
        evidence_blocks.append(f"[{i}] PMID:{pmid}\nTitle: {title}\nAbstract: {abstract}\n")

    prompt = f"""You are a careful biomedical assistant.
Answer the question using ONLY the evidence snippets below.
- If evidence is insufficient, say "Insufficient evidence from retrieved abstracts."
- Cite evidence using [1], [2], ... corresponding to the snippets.

Question:
{args.question}

Evidence snippets:
{chr(10).join(evidence_blocks)}

Now write:
1) A concise answer (3-6 sentences) with citations
2) A bullet list of key evidence points with citations
"""

    print("\n=== Answer (Ollama grounded) ===\n")
    try:
        out = _ollama_generate(prompt=prompt, model=args.ollama_model, host=args.ollama_host)
        print(out)
    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        print("Ollama가 실행 중인지 확인하세요: `ollama serve` / 모델 pull 여부 등.")


if __name__ == "__main__":
    main()
