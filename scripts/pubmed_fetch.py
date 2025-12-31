#!/usr/bin/env python3
"""
Fetch PubMed records via NCBI E-utilities and save as JSONL.

- Default behavior: keep ONLY records with non-empty abstracts.
- Output format matches build_index.py expectations: title, abstract, journal, pub_year, pmid, url

Example:
  python scripts/pubmed_fetch.py --query '"gamma delta T" CD16 function' --max_results 200 --out data/pubmed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _safe_int_year(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return m.group(1) if m else ""


def _esearch(query: str, retmax: int) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
    }
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", []) or []


def _efetch(pmids: List[str]) -> str:
    # EFetch can take many IDs; chunk outside for safety.
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []

    for art in root.findall(".//PubmedArticle"):
        pmid = ""
        pmid_node = art.find(".//MedlineCitation/PMID")
        if pmid_node is not None and pmid_node.text:
            pmid = pmid_node.text.strip()

        title = ""
        title_node = art.find(".//Article/ArticleTitle")
        if title_node is not None:
            title = _clean_text("".join(title_node.itertext()))

        # Abstract can be multiple <AbstractText> parts
        abs_parts: List[str] = []
        for abs_node in art.findall(".//Article/Abstract/AbstractText"):
            part = _clean_text("".join(abs_node.itertext()))
            if part:
                abs_parts.append(part)
        abstract = _clean_text(" ".join(abs_parts))

        journal = ""
        j_node = art.find(".//Article/Journal/Title")
        if j_node is not None and j_node.text:
            journal = _clean_text(j_node.text)

        year = ""
        # try PubDate Year first
        y_node = art.find(".//Article/Journal/JournalIssue/PubDate/Year")
        if y_node is not None and y_node.text:
            year = _safe_int_year(y_node.text)
        if not year:
            # fallback to MedlineDate (e.g., "1992 Jan-Feb")
            md_node = art.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
            if md_node is not None and md_node.text:
                year = _safe_int_year(md_node.text)

        rec = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        }
        out.append(rec)

    return out


def _ensure_outdir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed search query")
    ap.add_argument("--max_results", type=int, default=200, help="Max PMIDs to fetch")
    ap.add_argument("--out", default="data/pubmed.jsonl", help="Output JSONL")
    ap.add_argument(
        "--require_abstract",
        action="store_true",
        default=True,
        help="Keep only records with non-empty abstracts (default: true)",
    )
    ap.add_argument(
        "--min_abstract_len",
        type=int,
        default=50,
        help="Minimum abstract length to keep (default: 50 chars)",
    )
    ap.add_argument("--batch", type=int, default=100, help="EFetch batch size (default: 100)")
    ap.add_argument("--sleep", type=float, default=0.34, help="Sleep between batches (default: 0.34s)")
    args = ap.parse_args()

    _ensure_outdir(args.out)

    pmids = _esearch(args.query, args.max_results)
    if not pmids:
        print("No PMIDs found.")
        sys.exit(0)

    print(f"Found {len(pmids)} PMIDs. Fetching details...")

    kept = 0
    skipped_noabs = 0
    skipped_short = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(pmids), args.batch)):
            chunk = pmids[i : i + args.batch]
            xml_text = _efetch(chunk)
            recs = _parse_pubmed_xml(xml_text)

            for rec in recs:
                abstract = (rec.get("abstract") or "").strip()
                if args.require_abstract and not abstract:
                    skipped_noabs += 1
                    continue
                if args.require_abstract and len(abstract) < args.min_abstract_len:
                    skipped_short += 1
                    continue

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

            time.sleep(args.sleep)

    print(f"Saved {kept} records to: {args.out}")
    if args.require_abstract:
        print(f"Skipped (no abstract): {skipped_noabs}")
        print(f"Skipped (too short):  {skipped_short}")


if __name__ == "__main__":
    main()
