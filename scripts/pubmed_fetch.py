#!/usr/bin/env python3
"""
Fetch PubMed records (title/abstract/metadata) into JSONL.

Default behavior:
- Fetches up to --max_results PMIDs for --query, then downloads details in batches.
- Writes JSONL to --out.

Options:
- --require_abstract: keep only records with a non-empty abstract
- --min_abstract_chars: minimum abstract length (after joining parts)
- --allow_no_abstract: (legacy) allow missing abstracts (default: True unless --require_abstract)

Notes:
- Uses NCBI E-utilities (Entrez). You should set --email to something valid.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

USER_AGENT = "bio-rag-pubmed/1.0 (requests; python)"


def _clean_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _esearch_pmids(query: str, retmax: int) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
    }
    r = requests.get(ESEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _chunks(lst: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _extract_text(el) -> str:
    if el is None:
        return ""
    return _clean_ws(el.get_text(" ", strip=True))


def _parse_pub_year(article) -> str:
    # Try ArticleDate year, then JournalIssue PubDate year
    year = ""
    ad = article.find("ArticleDate")
    if ad and ad.find("Year"):
        year = _extract_text(ad.find("Year"))
    if not year:
        pubdate = article.find("JournalIssue")
        if pubdate:
            pd = pubdate.find("PubDate")
            if pd and pd.find("Year"):
                year = _extract_text(pd.find("Year"))
    return year


def _parse_abstract(article) -> str:
    """
    PubMed XML often has:
      <Abstract><AbstractText>...</AbstractText></Abstract>
    Sometimes multiple AbstractText nodes (with labels).
    """
    abs_el = article.find("Abstract")
    if not abs_el:
        return ""

    parts = []
    for at in abs_el.find_all("AbstractText"):
        txt = _extract_text(at)
        if not txt:
            continue
        label = at.get("Label") or at.get("NlmCategory") or ""
        label = _clean_ws(label)
        if label and label.lower() not in ("unspecified",):
            parts.append(f"{label}: {txt}")
        else:
            parts.append(txt)

    abstract = "\n".join(parts).strip()
    return abstract


def _efetch_details(pmids: List[str], email: str) -> List[Dict[str, Any]]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email

    r = requests.get(EFETCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml-xml")
    out: List[Dict[str, Any]] = []

    for article in soup.find_all("PubmedArticle"):
        medline = article.find("MedlineCitation")
        if not medline:
            continue

        pmid = _extract_text(medline.find("PMID"))
        art = medline.find("Article")
        if not art:
            continue

        title = _extract_text(art.find("ArticleTitle"))
        abstract = _parse_abstract(art)

        journal = ""
        j = art.find("Journal")
        if j and j.find("Title"):
            journal = _extract_text(j.find("Title"))

        pub_year = _parse_pub_year(art)

        rec: Dict[str, Any] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_year": pub_year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        }
        out.append(rec)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed query")
    ap.add_argument("--max_results", type=int, default=50, help="Max number of PMIDs to fetch")
    ap.add_argument("--out", default="data/pubmed.jsonl", help="Output JSONL path")
    ap.add_argument("--batch_size", type=int, default=200, help="EFetch batch size (PMIDs per request)")
    ap.add_argument("--email", default="", help="NCBI recommended: your email")
    ap.add_argument(
        "--allow_no_abstract",
        action="store_true",
        help="(legacy) Allow records missing abstracts (default behavior unless --require_abstract).",
    )
    ap.add_argument(
        "--require_abstract",
        action="store_true",
        help="Keep only records with non-empty abstracts.",
    )
    ap.add_argument(
        "--min_abstract_chars",
        type=int,
        default=0,
        help="Minimum abstract length (after joining parts). 0 = no minimum.",
    )
    args = ap.parse_args()

    pmids = _esearch_pmids(args.query, args.max_results)
    if not pmids:
        print("No PMIDs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pmids)} PMIDs. Fetching details...")

    records: List[Dict[str, Any]] = []
    for chunk in tqdm(list(_chunks(pmids, args.batch_size))):
        recs = _efetch_details(chunk, args.email)
        records.extend(recs)
        time.sleep(0.34)  # be polite

    # Filtering logic
    require_abs = args.require_abstract
    if require_abs:
        filtered = []
        for r in records:
            abs_txt = (r.get("abstract") or "").strip()
            if not abs_txt:
                continue
            if args.min_abstract_chars and len(abs_txt) < args.min_abstract_chars:
                continue
            filtered.append(r)
        records = filtered
    else:
        # Default/legacy:
        # If user did NOT request require_abstract, we keep everything.
        # (args.allow_no_abstract kept for backward compatibility; no-op here.)
        if args.min_abstract_chars:
            # if user sets min chars without require, apply it anyway (safe behavior)
            records = [
                r
                for r in records
                if (r.get("abstract") or "").strip()
                and len((r.get("abstract") or "").strip()) >= args.min_abstract_chars
            ]

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to: {args.out}")
    if require_abs and len(records) == 0:
        print(
            "WARNING: After filtering, 0 records remained. Try a broader query or reduce --min_abstract_chars.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
