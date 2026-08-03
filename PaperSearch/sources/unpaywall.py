"""
Thin client for the Unpaywall API — legal open-access full-text discovery.

Unpaywall aggregates *legally* available OA copies (publisher OA, repository
self-archiving, PMC, etc.) and never bypasses a paywall; if it has no OA
location for a DOI, that DOI is simply not downloadable here.
"""

from __future__ import annotations

from typing import Any, Optional

import requests

from .. import config

BASE = "https://api.unpaywall.org/v2"


def get_oa_pdf_url(doi: str) -> Optional[str]:
    """Return a legal open-access PDF URL for `doi`, if Unpaywall knows one, else None."""
    resp = requests.get(f"{BASE}/{doi}", params={"email": config.UNPAYWALL_EMAIL}, timeout=30)
    if resp.status_code == 404:
        return None
    if resp.status_code == 422:
        raise ValueError(
            f"Unpaywall rejected UNPAYWALL_EMAIL={config.UNPAYWALL_EMAIL!r} as invalid "
            "(it validates the email looks real). Set a real address in PaperSearch/.env."
        )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")