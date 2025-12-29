#!/usr/bin/env python3
"""
Fetch PubMed abstracts using NCBI E-utilities (Biopython Entrez)
and save as JSONL for downstream embedding/indexing.

Usage example:
  python scripts/pubmed_fetch.py --query "gamma delta T cells CD16" --max_results 200 --out data/pubmed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any, List

from Bio import Entrez
from Bio.Entrez import HTTPError
from tqdm import tqdm


def _configure_entrez(email: str | None, api_key: str | None) -> None:
    Entrez.email = email or os.getenv("NCBI_EMAIL") or "example@example.com"
    # NOTE: NCBI requests an email. Use your real email via env var NCBI_EMAIL.
    key = api_key or os.getenv("NCBI_API_KEY")
    if key:
        Entrez.api_key = key


def esearch_pmids(query: str, max_results: int) -> List[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    res = Entrez.read(handle)
    handle.close()
    return list(res.get("IdList", []))


def efetch_details(pmids: List[str], batch_size: int = 100) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        # efetch in XML
        tries = 0
        while True:
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    rettype="abstract",
                    retmode="xml",
                )
                data = Entrez.read(handle)
                handle.close()
                break
            except HTTPError as e:
                tries += 1
                if tries >= 5:
                    raise e
                time.sleep(1.5 * tries)

        articles = data.get("PubmedArticle", [])
        for art in articles:
            try:
                medline = art["MedlineCitation"]
                article = medline["Article"]

                pmid = str(medline["PMID"])
                title = str(article.get("ArticleTitle", "")).strip()

                # Abstract can be list of sections
                abstract_text = ""
                if "Abstract" in article and "AbstractText" in article["Abstract"]:
                    abs_parts = article["Abstract"]["AbstractText"]
                    # abs_parts elements can be strings or dict-like
                    joined = []
                    for p in abs_parts:
                        joined.append(str(p))
                    abstract_text = "\n".join(joined).strip()

                journal = ""
                if "Journal" in article and "Title" in article["Journal"]:
                    journal = str(article["Journal"]["Title"]).strip()

                year = ""
                if "Journal" in article and "JournalIssue" in article["Journal"]:
                    ji = article["Journal"]["JournalIssue"]
                    if "PubDate" in ji:
                        pd = ji["PubDate"]
                        year = str(pd.get("Year", pd.get("MedlineDate", ""))).strip()

                # DOI (optional)
                doi = ""
                if "ELocationID" in article:
                    for el in article["ELocationID"]:
                        try:
                            if el.attributes.get("EIdType") == "doi":
                                doi = str(el).strip()
                                break
                        except Exception:
                            pass

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "url": url,
                        "query_source": None,
                    }
                )
            except Exception:
                # skip malformed entries but continue
                continue

        # polite rate limit
        time.sleep(0.34)

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="PubMed query string")
    p.add_argument("--max_results", type=int, default=200, help="Max number of PMIDs to fetch")
    p.add_argument("--out", default="data/pubmed.jsonl", help="Output JSONL path")
    p.add_argument("--email", default=None, help="NCBI email (or set env NCBI_EMAIL)")
    p.add_argument("--api_key", default=None, help="NCBI API key (or set env NCBI_API_KEY)")
    p.add_argument("--batch_size", type=int, default=100, help="efetch batch size")
    args = p.parse_args()

    _configure_entrez(args.email, args.api_key)

    pmids = esearch_pmids(args.query, args.max_results)
    if not pmids:
        print("No PMIDs found. Try a different query.")
        return

    print(f"Found {len(pmids)} PMIDs. Fetching details...")
    # show progress by chunk count
    all_records: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(pmids), args.batch_size)):
        batch = pmids[i : i + args.batch_size]
        recs = efetch_details(batch, batch_size=len(batch))
        for r in recs:
            r["query_source"] = args.query
        all_records.extend(recs)

    # Ensure output dir exists (local run)
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_records:
            # keep only docs that have at least title or abstract
            if (r.get("title") or "").strip() or (r.get("abstract") or "").strip():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_records)} records to: {args.out}")


if __name__ == "__main__":
    main()
