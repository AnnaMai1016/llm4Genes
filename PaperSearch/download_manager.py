"""
Step 1: discover candidate papers via PubMed (NCBI Eutils) and CrossRef for
each configured query, resolve each DOI to a legally open-access full text
(PMC OA subset -> Unpaywall), download it, and record everything in the
manifest.

Papers with no reachable open-access full text are still recorded — metadata
only, `fulltext_status="no_legal_oa"` — so they can be retrieved manually
later rather than silently dropped.

Deliberately does NOT fall back to the institutional library proxy in this
batch path — see `resolve_fulltext_candidates` below and
`sources/institutional_proxy.py` for why (looping proxy requests over search
results is exactly the "systematic downloading" pattern library terms
prohibit, and publishers respond by cutting off proxied access for everyone
at the institution, not just one account). Use `fetch_via_institutional_proxy`
for one-off, human-triggered fetches of a specific paper instead.

Idempotent: re-running skips DOIs that already have a `pdf_path` in the manifest.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from curl_cffi import requests as cffi_requests

from . import config
from .manifest import find_row, load_manifest, save_manifest, upsert_row
from .sources import crossref, institutional_proxy, pubmed, scihub, unpaywall
from .utils import sanitize_filename

logger = logging.getLogger("papersearch.download_manager")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
}


def _fetch_bytes(url: str, cookies: Optional[dict[str, str]] = None) -> Optional[bytes]:
    """Fetch `url` and return raw bytes, or None if nothing worked.

    ftp:// (NCBI's OA mirror) goes through urllib. http/https tries plain
    `requests` first, then falls back to `curl_cffi` with a spoofed Chrome TLS
    fingerprint — a generic Python TLS handshake gets flat-out blocked by
    Cloudflare-protected publishers (MDPI etc.) regardless of headers, but a
    browser-shaped handshake clears that tier. It does NOT clear everything:
    sites behind heavier bot protection requiring real JS execution (Wiley,
    PeerJ) still block both — that's a real limitation, not a bug to chase
    further here (would require full browser automation).
    """
    if url.startswith("ftp://"):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError as exc:
            logger.warning("ftp fetch failed for %s: %s", url, exc)
            return None

    try:
        resp = requests.get(url, timeout=60, headers=_BROWSER_HEADERS, cookies=cookies)
        if resp.status_code == 200:
            return resp.content
        logger.info(
            "plain request got HTTP %d for %s; retrying with TLS-fingerprint spoofing",
            resp.status_code, url,
        )
    except requests.RequestException as exc:
        logger.info("plain request errored for %s (%s); retrying with TLS-fingerprint spoofing", url, exc)

    try:
        resp = cffi_requests.get(url, impersonate="chrome", timeout=60, headers=_BROWSER_HEADERS, cookies=cookies)
        if resp.status_code == 200:
            return resp.content
        logger.warning("curl_cffi (TLS-spoofed) also got HTTP %d for %s", resp.status_code, url)
    except Exception as exc:
        logger.warning("curl_cffi request failed for %s: %s", url, exc)
    return None


def _download_pdf(url: str, dest: Path, cookies: Optional[dict[str, str]] = None) -> bool:
    content = _fetch_bytes(url, cookies=cookies)
    if content is None:
        return False
    if content[:4] != b"%PDF":
        logger.warning("Downloaded content is not a PDF: %s", url)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return True


def resolve_fulltext_candidates(doi: str, pmid: Optional[str]) -> list[tuple[str, str]]:
    """Return (url, source_label) candidates in priority order: PMC OA, then
    Unpaywall. Both are tried in order — not just the first that resolves —
    because a resolved URL can still fail to *download* (NCBI's OA mirror
    paths are currently unreliable; some publisher sites block automated
    fetches even for genuinely open-access articles), so a second source is
    worth keeping as a fallback rather than giving up after the first hit.

    Deliberately does NOT include the institutional-proxy fallback: that one
    is a single-paper, human-triggered escape hatch only (see
    `fetch_via_institutional_proxy` below) — never looped over search results.
    UIUC's library proxy terms explicitly prohibit "systematic downloading...
    using robots, spiders or manual means", and publishers cut off proxied
    access institution-wide (everyone at the school, not just one account) if
    they detect that pattern. Wiring it into this per-DOI batch resolver would
    make exactly that pattern the default the moment someone flips the
    INSTITUTIONAL_PROXY_ENABLED switch — so it stays structurally out of the
    bulk path instead of relying on a comment to stop that from happening.
    """
    candidates: list[tuple[str, str]] = []
    if pmid:
        pmcid = pubmed.pubmed_to_pmcid(pmid)
        if pmcid:
            pdf_url = pubmed.pmc_oa_pdf_url(pmcid)
            if pdf_url:
                candidates.append((pdf_url, "pmc_oa"))
    if doi:
        pdf_url = unpaywall.get_oa_pdf_url(doi)
        if pdf_url:
            candidates.append((pdf_url, "unpaywall_oa"))
    return candidates


def fetch_via_institutional_proxy(doi: str) -> Optional[Path]:
    """Manually fetch ONE specific paper through your institutional library
    proxy. Call this yourself, one DOI at a time, for a paper you are
    personally reading right now — never in a loop, never from `run()` /
    `run_daily.py`. See the module docstring in `sources/institutional_proxy.py`
    and `resolve_fulltext_candidates` above for why this is kept separate.

    Returns the downloaded file path, or None if it couldn't be fetched.
    """
    if not institutional_proxy.available():
        problems = config.validate_for_institutional_proxy()
        raise RuntimeError("Institutional proxy not configured: " + "; ".join(problems))

    url = institutional_proxy.proxied_url(f"https://doi.org/{doi}")
    cookies = institutional_proxy.load_cookies()
    dest = config.PDF_DIR / f"{sanitize_filename(doi)}.pdf"
    if _download_pdf(url, dest, cookies=cookies):
        logger.info("fetch_via_institutional_proxy: downloaded doi=%s -> %s", doi, dest)
        return dest
    logger.warning("fetch_via_institutional_proxy: failed for doi=%s (url=%s)", doi, url)
    return None


def fetch_via_scihub(doi: str) -> Optional[Path]:
    """Manually fetch ONE specific paper via Sci-Hub. Call this yourself, one
    DOI at a time, for a paper you are personally reading right now — never
    in a loop, never from `run()` / `run_daily.py`. See
    `sources/scihub.py` and `resolve_fulltext_candidates` above for why: this
    is riskier than the institutional-proxy fallback (retrieves copies hosted
    without publisher authorization — copyright infringement, not just a ToS
    issue), so it stays structurally out of the bulk path too.

    Returns the downloaded file path, or None if it couldn't be fetched.
    """
    if not config.SCIHUB_ENABLED:
        raise RuntimeError("Sci-Hub fallback disabled: set SCIHUB_ENABLED=true in .env to use it.")

    dest = scihub.download(doi, config.PDF_DIR)
    if dest:
        logger.info("fetch_via_scihub: downloaded doi=%s -> %s", doi, dest)
        return dest
    logger.warning("fetch_via_scihub: failed for doi=%s", doi)
    return None


_DOI_PATTERN = re.compile(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b')


def _extract_doi_from_pdf(pdf_path: Path, max_pages: int = 2) -> Optional[str]:
    """Best-effort: pull a DOI out of a PDF's own first few pages of text.

    Publishers almost always print the DOI somewhere on page 1 (header,
    footer, or citation block), so this only reads `max_pages` rather than
    the whole file. Returns the first DOI-shaped regex match, or None if
    nothing was found or the PDF couldn't be read.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages[:max_pages])
    except Exception:
        logger.exception("failed to read %s while looking for a DOI", pdf_path)
        return None
    match = _DOI_PATTERN.search(text)
    return match.group(0).rstrip('.,;)') if match else None


def rename_pdfs_to_doi(dest_dir: Optional[Path] = None) -> list[tuple[Path, Path]]:
    """Rename PDFs in `dest_dir` (default `config.PDF_DIR`) to match the
    `<sanitize_filename(doi)>.pdf` convention the rest of the pipeline
    expects (`_process_hit`, `reconcile_manual_downloads`,
    `convert_mineru.py`), for files you saved by hand under their original
    publisher filename.

    For each file that isn't already named that way, extracts a DOI from the
    PDF's own text (see `_extract_doi_from_pdf`) and, only if that DOI
    matches a row already in the manifest, renames the file to
    `<sanitize_filename(doi)>.pdf`. Skips (and logs) anything where no DOI
    could be extracted or the extracted DOI isn't in the manifest — it never
    guesses from the filename itself.

    Does not touch the manifest — run `reconcile_manual_downloads()`
    afterwards to have the renamed files picked up there.

    Returns the list of (old_path, new_path) actually renamed.
    """
    dest_dir = dest_dir or config.PDF_DIR
    rows = load_manifest(config.MANIFEST_PATH)
    known_dois = {row["doi"] for row in rows if row.get("doi")}
    expected_names = {sanitize_filename(doi) + ".pdf" for doi in known_dois}
    dois_by_lower = {doi.lower(): doi for doi in known_dois}

    renamed: list[tuple[Path, Path]] = []
    for pdf_path in dest_dir.glob("*.pdf"):
        if pdf_path.name in expected_names:
            continue  # already correctly named

        found_doi = _extract_doi_from_pdf(pdf_path)
        matched_doi = dois_by_lower.get(found_doi.lower()) if found_doi else None
        if not matched_doi:
            logger.info(
                "rename_pdfs_to_doi: no manifest match for %s (extracted doi: %r)",
                pdf_path.name, found_doi,
            )
            continue

        new_path = dest_dir / f"{sanitize_filename(matched_doi)}.pdf"
        if new_path.exists():
            logger.warning("rename_pdfs_to_doi: target %s already exists, skipping %s", new_path.name, pdf_path.name)
            continue

        pdf_path.rename(new_path)
        logger.info("rename_pdfs_to_doi: %s -> %s", pdf_path.name, new_path.name)
        renamed.append((pdf_path, new_path))

    return renamed


def reconcile_manual_downloads() -> list[dict[str, Any]]:
    """Pick up PDFs you dropped into `config.PDF_DIR` by hand.

    For every manifest row that doesn't already have a `pdf_path`, check
    whether a file named `<sanitize_filename(doi)>.pdf` exists in
    `config.PDF_DIR` (the same naming convention `_process_hit` uses for
    automated downloads, and `convert_mineru.py` expects downstream). If it's
    there, mark the row `fulltext_status="download_manual"` and fill in
    `pdf_path` / `downloaded_at_utc`. If it's still not there, mark it
    `fulltext_status="need_to_be_download"` so it's visibly flagged as
    "waiting on you" instead of silently staying `download_failed`.

    This never fetches anything itself — it only reconciles what's already on
    disk — so it's safe to call as often as you like. Always resaves the full
    manifest so `need_to_be_download` rows are visible even if nothing changed.
    """
    rows = load_manifest(config.MANIFEST_PATH)
    touched: list[dict[str, Any]] = []
    for row in rows:
        doi = row.get("doi")
        if not doi or row.get("pdf_path"):
            continue
        pdf_path = config.PDF_DIR / f"{sanitize_filename(doi)}.pdf"
        if pdf_path.is_file():
            row["fulltext_status"] = "download_manual"
            row["pdf_path"] = str(pdf_path)
            row["downloaded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info("reconcile_manual_downloads: found manual pdf | doi=%s | %s", doi, pdf_path)
        else:
            row["fulltext_status"] = "need_to_be_download"
        touched.append(row)
    save_manifest(config.MANIFEST_PATH, rows)
    return touched


def _process_hit(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    query: str,
    seen_this_run: set[str],
) -> Optional[dict[str, Any]]:
    doi = summary.get("doi")
    if not doi:
        return None
    doi_key = doi.lower()
    if doi_key in seen_this_run:
        return None  # PubMed and CrossRef both matched this DOI in this run; don't re-resolve
    seen_this_run.add(doi_key)

    existing = find_row(rows, doi)
    if existing and existing.get("pdf_path"):
        return None  # already downloaded in a previous run

    row = upsert_row(
        rows,
        doi,
        {
            "pmid": summary.get("pmid") or (existing or {}).get("pmid"),
            "title": summary.get("title") or (existing or {}).get("title"),
            "pubdate": summary.get("pubdate") or (existing or {}).get("pubdate"),
            "source_journal": summary.get("source_journal") or (existing or {}).get("source_journal"),
            "matched_query": query,
        },
    )

    candidates = resolve_fulltext_candidates(doi, summary.get("pmid"))
    if not candidates:
        row["fulltext_status"] = "no_legal_oa"
        return row

    paper_id = sanitize_filename(doi)
    dest = config.PDF_DIR / f"{paper_id}.pdf"
    for pdf_url, source_label in candidates:
        if _download_pdf(pdf_url, dest):
            row["fulltext_status"] = source_label
            row["pdf_path"] = str(dest)
            row["downloaded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            row.pop("attempted_url", None)
            row.pop("attempted_urls", None)
            logger.info("downloaded | doi=%s | status=%s", doi, source_label)
            break
    else:
        # Every candidate resolved to a *legal* OA location but none could actually
        # be fetched (publisher anti-bot 403, or NCBI's OA mirror 404ing despite
        # oa.fcgi listing the file). Save every attempted URL for a manual grab.
        row["fulltext_status"] = "download_failed"
        row["attempted_urls"] = [u for u, _ in candidates]
        row["attempted_url"] = candidates[-1][0]

    return row


def run(queries: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Run one search+download pass over `queries` (default: config.SEARCH_QUERIES).

    Returns the manifest rows touched this run (newly seen or re-checked DOIs).
    """
    problems = config.validate_for_search()
    if problems:
        raise RuntimeError("PaperSearch config incomplete: " + "; ".join(problems))

    queries = queries if queries is not None else config.SEARCH_QUERIES
    rows = load_manifest(config.MANIFEST_PATH)
    touched: list[dict[str, Any]] = []
    seen_this_run: set[str] = set()

    from_pub_date: Optional[str] = None
    if config.SEARCH_RECENT_DAYS:
        from_pub_date = (date.today() - timedelta(days=config.SEARCH_RECENT_DAYS)).isoformat()

    for query in queries:
        logger.info("searching PubMed | query=%r", query)
        pmids = pubmed.esearch(
            query, retmax=config.MAX_RESULTS_PER_QUERY, reldate_days=config.SEARCH_RECENT_DAYS
        )
        logger.info("query %r -> %d PubMed hit(s)", query, len(pmids))
        for batch_start in range(0, len(pmids), 100):
            batch = pmids[batch_start: batch_start + 100]
            for summary in pubmed.esummary(batch):
                row = _process_hit(rows, summary, query, seen_this_run)
                if row is not None:
                    touched.append(row)
                    save_manifest(config.MANIFEST_PATH, rows)  # persist incrementally

        logger.info("searching CrossRef | query=%r", query)
        cr_items = crossref.search(query, rows=config.MAX_RESULTS_PER_QUERY, from_pub_date=from_pub_date)
        logger.info("query %r -> %d CrossRef hit(s)", query, len(cr_items))
        for item in cr_items:
            summary = crossref.to_manifest_row(item)
            row = _process_hit(rows, summary, query, seen_this_run)
            if row is not None:
                touched.append(row)
                save_manifest(config.MANIFEST_PATH, rows)

    return touched


def main():
    parser = argparse.ArgumentParser(prog="download_manager")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Pick up PDFs you saved by hand into config.PDF_DIR (rename them "
             "to the <doi>.pdf convention, then sync fulltext_status in the "
             "manifest), instead of running the automated search+download pass.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.reconcile:
        renamed = rename_pdfs_to_doi()
        for old_path, new_path in renamed:
            print(f"renamed: {old_path.name} -> {new_path.name}")
        touched = reconcile_manual_downloads()
        print(f"reconciled {len(touched)} row(s); manifest at {config.MANIFEST_PATH}")
        return

    new_rows = run()
    print(f"processed {len(new_rows)} paper(s); manifest at {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()