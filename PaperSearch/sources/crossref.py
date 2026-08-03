"""Thin client for the CrossRef REST API (DOI metadata lookup / search)."""

from __future__ import annotations

from typing import Any, Optional

import requests

from .. import config

BASE = "https://api.crossref.org/works"


def lookup_by_doi(doi: str) -> Optional[dict[str, Any]]:
    resp = requests.get(f"{BASE}/{doi}", params={"mailto": config.CROSSREF_MAILTO}, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("message")


def search(query: str, rows: int = 20, from_pub_date: Optional[str] = None) -> list[dict[str, Any]]:
    """`from_pub_date`: 'YYYY-MM-DD' string; restricts to works published on/after that date."""
    params = {"query": query, "rows": rows, "mailto": config.CROSSREF_MAILTO}
    if from_pub_date:
        params["filter"] = f"from-pub-date:{from_pub_date}"
    resp = requests.get(BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("items", [])


def to_manifest_row(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a CrossRef work item to the same shape as `pubmed.esummary()` rows."""
    titles = item.get("title") or []
    date_block = item.get("published") or item.get("published-print") or item.get("issued") or {}
    date_parts = date_block.get("date-parts") or [[]]
    pubdate = "-".join(str(p) for p in date_parts[0]) if date_parts and date_parts[0] else None
    journals = item.get("container-title") or []
    return {
        "pmid": None,
        "doi": item.get("DOI"),
        "title": titles[0] if titles else None,
        "pubdate": pubdate,
        "source_journal": journals[0] if journals else None,
    }