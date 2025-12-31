#!/usr/bin/env python3
"""
Fetch PubMed records (title/abstract/metadata) into JSONL.

Key behavior:
- By default, forces PubMed search to return ONLY records that have abstracts,
  using 'hasabstract[text]' at the search stage.
- Also filters again at save stage (if abstract missing -> skip).

Output JSONL fields:
  pmid, title, abstract, journal, pub_year, url
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests
from xml.etree import ElementTree as ET
from tqdm import tqdm

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _request(url: str, params: Dict[str, Any], retries: int = 5, backoff: float = 1.5) -> requests.Response:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep((backoff ** i) + 0.2)
    raise RuntimeError(f"Request failed after {retries} retries: {url} params={params} err={last_err}")

def _esearch(query: str, retmax: int, email: str = "", tool: str = "bio-rag-pubmed") -> List[str]:
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "tool": tool,
    }
    if email:
        params["email"] = email

    r = _request(url, params)
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", []) or []

def _chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _safe_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    # join text of element and all subelements
    return "".join(el.itertext()).strip()

def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []

    for art in root.findall(".//PubmedArticle"):
        pmid = _safe_text(art.find(".//MedlineCitation/PMID"))
        title = _safe_text(art.find(".//Article/ArticleTitle"))

        # Abstract can be multiple <AbstractText> blocks
        abs_texts = []
        for at in art.findall(".//Article/Abstract/AbstractText"):
            t = _safe_text(at)
            if t:
                abs_texts.append(t)
        abstract = "\n".join(abs_texts).strip()

        journal = _safe_text(art.find(".//Article/Journal/Title"))

        # Year: try PubDate/Year first, fallback to MedlineDate (extract first 4 digits)
        year = _safe_text(art.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline_date = _safe_text(art.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
            if medline_date:
                # crude: take first 4-digit chunk
                for token in medline_date.replace("-", " ").split():
                    if len(token) == 4 and token.isdigit():
                        year = token
                        break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        out.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_year": year,
                "url": url,
            }
        )
    return out

def _efetch(pmids: List[str], email: str = "", tool: str = "bio-rag-pubmed") -> List[Dict[str, Any]]:
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
    }
    if email:
        params["email"] = email

    r = _request(url, params)
    return _parse_pubmed_xml(r.text)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed query term (without hasabstract filter)")
    ap.add_argument("--max_results", type=int, default=100)
    ap.add_argument("--out", default="data/pubmed.jsonl")
    ap.add_argument("--batch_size", type=int, default=100, help="PMIDs per efetch batch")
    ap.add_argument("--email", default="", help="NCBI recommends an email for E-utilities")
    ap.add_argument("--allow_no_abstract", action="store_true", help="If set, do NOT filter by abstract")
    args = ap.parse_args()

    # Force hasabstract at search stage unless user explicitly allows no-abstract
    term = args.query.strip()
    if not args.allow_no_abstract:
        term = f"({term}) AND hasabstract[text]"

    pmids = _esearch(term, retmax=args.max_results, email=args.email)
    print(f"Found {len(pmids)} PMIDs. Fetching details...")

    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in tqdm(_chunks(pmids, args.batch_size)):
            recs = _efetch(chunk, email=args.email)
            for rec in recs:
                # final guard: skip empty abstract unless allow_no_abstract
                if (not args.allow_no_abstract) and (not (rec.get("abstract") or "").strip()):
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Saved {kept} records to: {args.out}")
    if not args.allow_no_abstract and kept < len(pmids):
        print(f"Note: skipped {len(pmids) - kept} records without abstracts.")

if __name__ == "__main__":
    main()
