"""
PubMed fetcher (public data only)

- Search PubMed with a query
- Fetch abstracts for returned PMIDs
- Save as JSONL for downstream indexing

Usage (later, from terminal):
  python scripts/pubmed_fetch.py --query "gamma delta T cell CD16" --max 50 --out data/pubmed.jsonl

NCBI policy: set your email (required by Entrez).
  export NCBI_EMAIL="you@example.com"
Optional:
  export NCBI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

from Bio import Entrez
from tqdm import tqdm


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    journal: str
    year: str
    authors: List[str]
    abstract: str
    url: str


def _configure_entrez() -> None:
    email = os.getenv("NCBI_EMAIL")
    if not email:
        raise RuntimeError(
            "NCBI_EMAIL is not set. Please set it, e.g.\n"
            '  export NCBI_EMAIL="you@example.com"\n'
            "(Entrez requires an email address.)"
        )
    Entrez.email = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def search_pmids(query: str, max_results: int = 100) -> List[str]:
    """Return a list of PMIDs for the given PubMed query."""
    _configure_entrez()
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    results = Entrez.read(handle)
    handle.close()
    return list(results.get("IdList", []))


def fetch_details(pmids: Iterable[str], sleep_sec: float = 0.34) -> List[PubMedRecord]:
    """
    Fetch article details/abstracts for PMIDs using efetch (XML).
    sleep_sec: be polite to NCBI; with API key you can go faster.
    """
    _configure_entrez()
    pmid_list = list(pmids)
    if not pmid_list:
        return []

    records: List[PubMedRecord] = []

    # Fetch in chunks (NCBI-friendly)
    chunk_size = 50
    for i in range(0, len(pmid_list), chunk_size):
        chunk = pmid_list[i : i + chunk_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(chunk), retmode="xml")
        data = Entrez.read(handle)
        handle.close()

        articles = data.get("PubmedArticle", [])
        for art in articles:
            medline = art.get("MedlineCitation", {})
            article = medline.get("Article", {})

            pmid = str(medline.get("PMID", ""))

            title = " ".join(str(article.get("ArticleTitle", "")).split())
            journal = ""
            year = ""

            journal_info = article.get("Journal", {})
            journal = str(journal_info.get("Title", "") or "")

            # Year
            journal_issue = journal_info.get("JournalIssue", {})
            pub_date = journal_issue.get("PubDate", {})
            year = str(pub_date.get("Year", "") or "")

            # Authors
            authors = []
            for a in article.get("AuthorList", []) or []:
                last = a.get("LastName")
                fore = a.get("ForeName")
                if last and fore:
                    authors.append(f"{fore} {last}")
                elif last:
                    authors.append(str(last))

            # Abstract (may have multiple sections)
            abstract_text = ""
            abstract = article.get("Abstract")
            if abstract and "AbstractText" in abstract:
                parts = abstract.get("AbstractText", [])
                # parts can contain strings or dict-like with labels
                texts = []
                for p in parts:
                    texts.append(str(p))
                abstract_text = "\n".join(t.strip() for t in texts if t.strip())

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            # Keep only useful records (must have abstract)
            if pmid and abstract_text:
                records.append(
                    PubMedRecord(
                        pmid=pmid,
                        title=title,
                        journal=journal,
                        year=year,
                        authors=authors,
                        abstract=abstract_text,
                        url=url,
                    )
                )

        time.sleep(sleep_sec)

    return records


def save_jsonl(records: List[PubMedRecord], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="PubMed query string")
    p.add_argument("--max", type=int, default=50, help="Max number of PMIDs to fetch")
    p.add_argument("--out", default="data/pubmed.jsonl", help="Output JSONL path")
    args = p.parse_args()

    pmids = search_pmids(args.query, max_results=args.max)
    if not pmids:
        print("No PMIDs found.")
        return

    recs = []
    for r in tqdm(fetch_details(pmids), desc="Fetching abstracts"):
        recs.append(r)

    save_jsonl(recs, args.out)
    print(f"Saved {len(recs)} records to {args.out}")


if __name__ == "__main__":
    main()
