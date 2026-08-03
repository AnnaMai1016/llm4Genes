"""
Shared JSON-manifest helpers for the PaperSearch pipeline.

The manifest is a single JSON array (`config.MANIFEST_PATH`), one row per
paper, keyed by DOI. Each stage (search/download, MinerU convert) updates its
own fields on a row rather than owning a separate file, so a paper's full
local history stays in one place:

    {"doi": ..., "pmid": ..., "title": ..., "matched_query": ...,
     "fulltext_status": "pmc_oa" | "unpaywall_oa" | "no_legal_oa" | "download_failed",
     "pdf_path": ..., "downloaded_at_utc": ...,
     "mineru_status": "done" | "failed" | "missing_pdf", "markdown_path": ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)


def find_row(rows: list[dict[str, Any]], doi: str) -> Optional[dict[str, Any]]:
    doi_lower = doi.lower()
    for row in rows:
        if str(row.get("doi", "")).lower() == doi_lower:
            return row
    return None


def upsert_row(rows: list[dict[str, Any]], doi: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge `updates` into the row for `doi`, creating it (appended to `rows`) if absent."""
    row = find_row(rows, doi)
    if row is None:
        row = {"doi": doi}
        rows.append(row)
    row.update(updates)
    return row