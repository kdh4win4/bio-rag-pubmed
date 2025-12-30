# src/rag.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    year: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None


class BioRAG:
    """
    Minimal retrieval engine:
      Question -> embed -> FAISS search -> TopK Paper objects

    Expects build_index.py to have produced:
      - data/index.faiss
      - data/meta.jsonl   (1 line per doc, aligned with FAISS ids: 0..N-1)

    meta.jsonl each line should include at least:
      {"pmid": "...", "title": "...", "abstract": "...", "year": "...", "journal": "...", "url": "..."}
    """

    def __init__(
        self,
        index: faiss.Index,
        meta: List[Dict[str, Any]],
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        self.index = index
        self.meta = meta
        self.normalize = normalize
        self.model = SentenceTransformer(embed_model)

    @staticmethod
    def load(
        index_path: reminder := str | Path,
        meta_path: str | Path,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> "BioRAG":
        index_path = Path(index_path)
        meta_path = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        index = faiss.read_index(str(index_path))

        meta: List[Dict[str, Any]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta.append(json.loads(line))

        # Quick sanity check
        if index.ntotal != len(meta):
            raise ValueError(
                f"Index/meta size mismatch: index.ntotal={index.ntotal} vs meta_lines={len(meta)}. "
                "build_index.py must write meta.jsonl in the exact same order as vectors were added."
            )

        return BioRAG(index=index, meta=meta, embed_model=embed_model, normalize=normalize)

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True).astype("float32")
        if self.normalize:
            faiss.normalize_L2(vec)
        return vec

    def retrieve(self, question: str, k: int = 5) -> List[Paper]:
        if k <= 0:
            return []

        q = self._embed(question)
        scores, ids = self.index.search(q, k)

        out: List[Paper] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            m = self.meta[idx]
            out.append(
                Paper(
                    pmid=str(m.get("pmid", "")),
                    title=str(m.get("title", "")),
                    abstract=str(m.get("abstract", "")),
                    year=(str(m.get("year")) if m.get("year") is not None else None),
                    journal=(str(m.get("journal")) if m.get("journal") is not None else None),
                    url=(str(m.get("url")) if m.get("url") is not None else None),
                    score=float(score),
                )
            )
        return out

    def format_evidence_block(self, papers: List[Paper]) -> str:
        """
        Returns a compact evidence text block you can feed into an LLM.
        """
        chunks: List[str] = []
        for i, p in enumerate(papers, start=1):
            header = f"[{i}] PMID:{p.pmid} | {p.year or ''} {p.journal or ''}".strip()
            title = p.title.strip()
            abstract = p.abstract.strip()
            url = f"URL: {p.url}" if p.url else ""
            chunks.append("\n".join(x for x in [header, title, abstract, url] if x))
        return "\n\n---\n\n".join(chunks)
