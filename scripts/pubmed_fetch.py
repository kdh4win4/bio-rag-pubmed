#!/usr/bin/env python3
"""
Fetch PubMed records into JSONL using NCBI E-utilities.

Output JSONL fields (per line):
{
  "pmid": "...",
  "title": "...",
  "abstract": "...",
  "journal": "...",
  "pub_year": "YYYY",
  "authors": ["Last F", ...],
  "url": "https://pubmed.ncbi.nlm.nih.gov/<PMID>/"
}

Key options:
- --require_abstract: drop records with missing/empty abstracts
- --min_abstract_chars: drop records with abstract shorter than N characters
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _safe_text(x: Optional[str]) -> str:
    return (x or "").strip()


def _sleep_polite():
    # NCBI etiquette: keep it gentle
    time.sleep(0.34)


def _soup_xml(xml_text: str) -> BeautifulSoup:
    """
    Prefer lxml-xml if available; otherwise fall back to built-in xml parser.
    """
    try:
        return BeautifulSoup(xml_text, "lxml-xml")
    except Exception:
        return BeautifulSoup(xml_text, "xml")


def _esearch_pmids(query: str, retmax: int) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
    }
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", []) or []


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _extract_year(article_tag) -> str:
    # Try PubDate/Year first, then MedlineDate fallback
    year = ""
    pubdate = article_tag.find("PubDate")
    if pubdate:
        y = pubdate.find("Year")
        if y and y.text:
            year = y.text.strip()
        else:
            md = pubdate.find("MedlineDate")
            if md and md.text:
                m = re.search(r"(19|20)\d{2}", md.text)
                if m:
                    year = m.group(0)
    return year


def _extract_abstract(article_tag) -> str:
    abs_tag = article_tag.find("Abstract")
    if not abs_tag:
        return ""
    parts = []
    for t in abs_tag.find_all("AbstractText"):
        txt = _safe_text(t.get_text(" ", strip=True))
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def _extract_title(article_tag) -> str:
    title_tag = article_tag.find("ArticleTitle")
    if not title_tag:
        return ""
    return _safe_text(title_tag.get_text(" ", strip=True))


def _extract_journal(article_tag) -> str:
    j = article_tag.find("Journal")
    if not j:
        return ""
    jt = j.find("Title")
    if jt and jt.text:
        return jt.text.strip()
    iso = j.find("ISOAbbreviation")
    if iso and iso.text:
        return iso.text.strip()
    return ""


def _extract_authors(article_tag) -> List[str]:
    out = []
    auth_list = article_tag.find("AuthorList")
    if not auth_list:
        return out
    for a in auth_list.find_all("Author"):
        last = a.find("LastName")
        fore = a.find("ForeName")
        if last and last.text:
            name = last.text.strip()
            if fore and fore.text:
                name = f"{name} {fore.text.strip()}"
            out.append(name)
    return out


def _efetch_details(pmids: List[str], email: Optional[str]) -> List[Dict[str, Any]]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email

    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    soup = _soup_xml(r.text)

    recs: List[Dict[str, Any]] = []
    for pubmed_article in soup.find_all("PubmedArticle"):
        pmid_tag = pubmed_article.find("PMID")
        article = pubmed_article.find("Article")
        if not pmid_tag or not article:
            continue

        pmid = _safe_text(pmid_tag.text)
        title = _extract_title(article)
        abstract = _extract_abstract(article)
        journal = _extract_journal(article)
        year = _extract_year(article)
        authors = _extract_authors(article)

        recs.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_year": year,
                "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return recs


def _passes_filters(rec: Dict[str, Any], require_abstract: bool, min_abs_chars: int) -> bool:
    abs_txt = _safe_text(rec.get("abstract"))
    if require_abstract and not abs_txt:
        return False
    if min_abs_chars > 0 and len(abs_txt) < min_abs_chars:
        return False
    # also drop if title is empty (rare but happens)
    if not _safe_text(rec.get("title")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed query string")
    ap.add_argument("--max_results", type=int, default=50)
    ap.add_argument("--out", default="data/pubmed.jsonl")
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""), help="NCBI courtesy email (optional)")
    ap.add_argument("--allow_no_abstract", action="store_true", help="Back-compat: allow saving records without abstracts")
    # NEW filters
    ap.add_argument("--require_abstract", action="store_true", help="Save only records that have an abstract")
    ap.add_argument("--min_abstract_chars", type=int, default=0, help="Minimum abstract length (characters) to keep")
    args = ap.parse_args()

    # Back-compat behavior:
    # If user passed --allow_no_abstract, we do NOT require abstract.
    require_abstract = args.require_abstract and (not args.allow_no_abstract)

    pmids = _esearch_pmids(args.query, args.max_results)
    print(f"Found {len(pmids)} PMIDs. Fetching details...")

    kept: List[Dict[str, Any]] = []
    for chunk in tqdm(_chunk(pmids, args.batch_size), total=max(1, (len(pmids) + args.batch_size - 1) // args.batch_size)):
        recs = _efetch_details(chunk, args.email or None)
        for rec in recs:
            if _passes_filters(rec, require_abstract=require_abstract, min_abs_chars=args.min_abstract_chars):
                kept.append(rec)
        _sleep_polite()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dropped = len(pmids) - len(kept)
    print(f"Saved {len(kept)} records to: {args.out}")
    if require_abstract or args.min_abstract_chars > 0:
        print(f"Filtered out {dropped} records (missing/short abstracts or missing titles).")


if __name__ == "__main__":
    main()
