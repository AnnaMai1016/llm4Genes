"""
Thin client for NCBI Entrez E-utilities: PubMed search + PMC open-access full text.

Usage policy: https://www.ncbi.nlm.nih.gov/books/NBK25497/ — every request
identifies a contact email; requests are capped at 3/sec without an API key,
10/sec with one (see `config.NCBI_API_KEY`).

`pmc_oa_pdf_url` only returns a link for articles in the PMC *Open Access*
subset (legally bulk-downloadable). Articles that are merely free-to-read on
the PMC website but not in that subset correctly return nothing here — that
distinction is enforced by NCBI itself, not worked around.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests

from .. import config

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

_last_request_time = 0.0


def _throttle() -> None:
    global _last_request_time
    min_interval = 0.11 if config.NCBI_API_KEY else 0.35
    elapsed = time.monotonic() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.monotonic()


def _params(extra: dict[str, Any]) -> dict[str, Any]:
    p = {"email": config.ENTREZ_EMAIL, **extra}
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    return p


def esearch(query: str, retmax: int = 100, reldate_days: Optional[int] = None) -> list[str]:
    """Return PubMed IDs (PMIDs) matching `query`."""
    params = _params({"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"})
    if reldate_days:
        params["reldate"] = reldate_days
        params["datetype"] = "pdat"
    _throttle()
    resp = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def esummary(pmids: list[str]) -> list[dict[str, Any]]:
    """Return basic metadata (title, DOI, pubdate, journal) for a batch of PMIDs."""
    if not pmids:
        return []
    params = _params({"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    _throttle()
    resp = requests.get(f"{BASE}/esummary.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    result = resp.json().get("result", {})
    rows = []
    for uid in result.get("uids", []):
        doc = result.get(uid, {})
        doi = next(
            (aid.get("value") for aid in doc.get("articleids", []) if aid.get("idtype") == "doi"),
            None,
        )
        rows.append(
            {
                "pmid": uid,
                "doi": doi,
                "title": doc.get("title"),
                "pubdate": doc.get("pubdate"),
                "source_journal": doc.get("fulljournalname") or doc.get("source"),
            }
        )
    return rows


def pubmed_to_pmcid(pmid: str) -> Optional[str]:
    """Return the linked PMC ID for a PMID, if the article has one, else None."""
    params = _params({"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"})
    _throttle()
    resp = requests.get(f"{BASE}/elink.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    for linkset in resp.json().get("linksets", []):
        for linksetdb in linkset.get("linksetdbs", []):
            links = linksetdb.get("links", [])
            if links:
                return f"PMC{links[0]}"
    return None


def pmc_oa_pdf_url(pmcid: str) -> Optional[str]:
    """
    Return a downloadable PDF URL for `pmcid` if it's in the PMC Open Access subset,
    else None. NCBI's OA service hands out `ftp://` links (its https mirror of the
    same path occasionally lags for very recently added records); the caller
    (`download_manager._download_pdf`) fetches `ftp://` URLs natively.
    """
    _throttle()
    resp = requests.get(OA_BASE, params={"id": pmcid}, timeout=30)
    resp.raise_for_status()
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None
    for link in root.iter("link"):
        if link.get("format") == "pdf" and link.get("href"):
            return link.get("href")
    return None