"""
Manual, single-paper fallback: fetch ONE specific paper from Sci-Hub, for an
article you are personally reading right now that had no legal open-access
route (PMC OA, Unpaywall) and no institutional-subscription access either.

READ THIS FIRST: unlike `institutional_proxy.py` (which replays your own
legitimate subscription session), Sci-Hub mirrors host copies of copyrighted
papers without publisher authorization — this is copyright infringement in
most jurisdictions, not merely a Terms-of-Service issue. That's why this
module is NOT wired into `download_manager.resolve_fulltext_candidates` /
`run()` / `run_daily.py`: never call `download_manager.fetch_via_scihub` in a
loop over search results or on a schedule. Use it by hand, one DOI at a time,
and only where you've personally decided that's an acceptable thing to do.

Off by default (`config.SCIHUB_ENABLED`).

Requires the `scihub-cn` package (installed in the `llmReview` conda env,
editable, from `PaperSearch/scihub-cn-master`):
https://github.com/Ckend/scihub-cn
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("papersearch.scihub")

_client = None  # lazily-built SciHub() instance; construction itself makes a
                 # blocking network call to fetch the current list of live mirrors


def _get_client():
    global _client
    if _client is None:
        from scihub_cn.scihub import SciHub
        _client = SciHub()
    return _client


def download(doi: str, dest_dir: Path) -> Optional[Path]:
    """Fetch `doi` via Sci-Hub and save it under `dest_dir`.

    Returns the saved file path, or None if no mirror had it / the fetch failed.
    """
    sh = _get_client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        paper_info = sh.download({"doi": doi}, destination=str(dest_dir))
    except Exception:
        logger.exception("scihub download raised for doi=%s", doi)
        return None
    if paper_info is None:
        return None
    return dest_dir / sh._vaild_name(paper_info.title)
