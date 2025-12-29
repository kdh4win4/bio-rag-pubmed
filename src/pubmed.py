"""
pubmed.py

Functions to fetch PubMed abstracts using NCBI Entrez.
Uses only public data.
"""

from typing import List
from Bio import Entrez


def setup_entrez(email: str, api_key: str | None = None) -> None:
    """
    Configure Entrez with user email (required by NCBI).
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def search_pubmed(query: str, max_results: int = 10) -> List[str]:
    """
    Search PubMed and return a list of PMIDs.
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmids: List[str]) -> List[dict]:
    """
    Fetch title and abstract text for a list of PMIDs.
    """
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()

    results = []

    for article in records["PubmedArticle"]:
        try:
            citation = article["MedlineCitation"]
            pmid = str(citation["PMID"])
            article_data = citation["Article"]

            title = str(article_data.get("ArticleTitle", ""))

            abstract = ""
            if "Abstract" in article_data:
                abstract_parts = article_data["Abstract"]["AbstractText"]
                abstract = " ".join([str(p) for p in abstract_parts])

            results.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                }
            )
        except Exception:
            continue

    return results
