"""
RAG core: load a FAISS index + metadata and perform retrieval for a user query.

This module expects:
- data/index.faiss            (FAISS index file)
- data/meta.jsonl             (JSONL with fields: pmid, title, year, journal, abstract, url)

You will create those files in later steps (build_index.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None  # type: ignore

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Hit:
    score: float
    pmid: str
    title: str
    year: str
    journal: str
    abstract: str
    url: str


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _normalize(v: np.ndarray) -> np.ndarray:
    # Safe L2 normalize for cosine similarity with IndexFlatIP
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


class BioRAG:
    """
    Minimal retrieval engine:
    - embeds the query
    - searches FAISS
    - returns top-k hits with PubMed links
    """

    def __init__(
        self,
        data_dir: str = "data",
        embedding_model: str = DEFAULT_MODEL,
        use_cosine: bool = True,
    ) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss import failed. Install with `pip install faiss-cpu`."
            )

        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir / "index.faiss"
        self.meta_path = self.data_dir / "meta.jsonl"

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing metadata jsonl: {self.meta_path}")

        self.model = SentenceTransformer(embedding_model)
        self.use_cosine = use_cosine

        self.index = faiss.read_index(str(self.index_path))
        self.meta = _read_jsonl(self.meta_path)

        # Sanity check: meta length should match index ntotal (most cases)
        try:
            ntotal = self.index.ntotal
            if ntotal != len(self.meta):
                # Not fatal, but warn via raising only if wildly off
                # (some indices may contain extra vectors)
                pass
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 5) -> List[Hit]:
        if not query.strip():
            return []

        q_emb = self.model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")

        # If the index was built for cosine similarity using inner product,
        # normalize vectors at query-time as well.
        if self.use_cosine:
            q_emb = _normalize(q_emb)

        scores, idxs = self.index.search(q_emb, top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        hits: List[Hit] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            if i >= len(self.meta):
                continue

            m = self.meta[i]
            pmid = str(m.get("pmid", "")).strip()
            title = (m.get("title") or "").strip()
            year = str(m.get("year", "")).strip()
            journal = (m.get("journal") or "").strip()
            abstract = (m.get("abstract") or "").strip()
            url = (m.get("url") or "").strip()
            if not url and pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            hits.append(
                Hit(
                    score=float(score),
                    pmid=pmid,
                    title=title,
                    year=year,
                    journal=journal,
                    abstract=abstract,
                    url=url,
                )
            )
        return hits

    @staticmethod
    def format_evidence(hits: List[Hit]) -> str:
        """
        Formats top-k retrieval results as a citation-friendly block.
        (LLM prompt에 그대로 붙여넣기 좋게 구성)
        """
        if not hits:
            return "No relevant evidence found."

        lines: List[str] = []
        for j, h in enumerate(hits, start=1):
            abs_short = h.abstract.replace("\n", " ").strip()
            if len(abs_short) > 600:
                abs_short = abs_short[:600].rstrip() + "…"

            lines.append(
                f"[{j}] PMID:{h.pmid} | {h.title} ({h.year}) {h.journal}\n"
                f"URL: {h.url}\n"
                f"Abstract: {abs_short}\n"
            )
        return "\n".join(lines)


def quick_demo() -> None:
    """
    Simple local demo (no LLM):
    - loads index
    - asks query
    - prints evidence block
    """
    rag = BioRAG(data_dir="data")
    q = input("Ask a bio question: ").strip()
    hits = rag.retrieve(q, top_k=5)
    print("\n=== Top evidence ===\n")
    print(BioRAG.format_evidence(hits))


if __name__ == "__main__":
    quick_demo()
