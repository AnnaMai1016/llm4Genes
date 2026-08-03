"""
Central configuration for the PaperSearch pipeline
(step 1: search/download literature, step 2: MinerU conversion).

Everything is environment-driven. Copy `.env.example` to `.env` in this same
directory and fill in real values — `.env` is loaded below and should never
be committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: List[str]) -> List[str]:
    val = os.environ.get(name)
    if not val:
        return default
    return [item.strip() for item in val.split("||") if item.strip()]


# -----------------------------
# Identity required by NCBI Eutils / CrossRef / Unpaywall usage policies.
# None of these are secrets, but each API asks every caller to self-identify.
# -----------------------------
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")
# Optional: raises the NCBI Eutils rate limit from 3 req/s to 10 req/s.
# https://www.ncbi.nlm.nih.gov/books/NBK25497/
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", ENTREZ_EMAIL)
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", ENTREZ_EMAIL)

# -----------------------------
# Search scope (SWEET / sugar-transporter genotype-phenotype focus for now).
# "||"-separated in the env var; results from every query are deduped by DOI.
# -----------------------------
SEARCH_QUERIES: List[str] = _list_env(
    "SEARCH_QUERIES",
    default=[
        "SWEET transporter gene phenotype",
        "SWEET family sugar transporter maize",
        "SWEET family sugar transporter rice",
    ],
)
MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "100"))
# Restrict PubMed search to papers indexed within the last N days (None = no limit).
# Set this once you're running the pipeline daily, so each run only looks at new papers.
_recent_days = os.environ.get("SEARCH_RECENT_DAYS", "")
SEARCH_RECENT_DAYS: Optional[int] = int(_recent_days) if _recent_days else None

# -----------------------------
# Storage locations
# -----------------------------
DATA_DIR = Path(os.environ.get("PAPERSEARCH_DATA_DIR") or (PROJECT_ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Step 2 output: multimodal Markdown produced by MinerU.
MINERU_OUTPUT_DIR = Path(os.environ.get("MINERU_OUTPUT_DIR") or (REPO_ROOT / "Project-Gene"))
MINERU_BIN = os.environ.get("MINERU_BIN", "mineru")
# "vlm" = vision-language-model backend -> multimodal parsing (figures/tables
# understood, not just OCR text). Matches the `<paper_id>/vlm/<paper_id>.md`
# layout PaperReviewTool/assessment already expects from MinerU.
MINERU_BACKEND = os.environ.get("MINERU_BACKEND", "vlm")

# -----------------------------
# Institutional library-proxy fallback (e.g. EZproxy) — off by default.
# For papers your school already has legitimate subscription access to; this
# replays your own logged-in browser session, it does not bypass anything.
# See sources/institutional_proxy.py for the one-time manual setup (find your
# library's proxy login URL, log in once in a real browser, export cookies).
# -----------------------------
INSTITUTIONAL_PROXY_ENABLED = _bool_env("INSTITUTIONAL_PROXY_ENABLED", default=False)
# Must contain a literal "{url}" placeholder, e.g.
# "https://libproxy.example.edu/login?url={url}" (standard EZproxy pattern).
INSTITUTIONAL_PROXY_URL_TEMPLATE = os.environ.get("INSTITUTIONAL_PROXY_URL_TEMPLATE", "")
# Netscape-format cookies.txt exported from your browser after logging into
# the URL above (e.g. via the "Get cookies.txt LOCALLY" browser extension).
INSTITUTIONAL_PROXY_COOKIES_FILE = Path(
    os.environ.get("INSTITUTIONAL_PROXY_COOKIES_FILE") or (DATA_DIR / "institutional_cookies.txt")
)

# -----------------------------
# Sci-Hub fallback — off by default, and never wired into the batch run().
# See sources/scihub.py: unlike the institutional-proxy fallback above, this
# retrieves copies hosted without publisher authorization. Manual,
# one-DOI-at-a-time use only.
# -----------------------------
SCIHUB_ENABLED = _bool_env("SCIHUB_ENABLED", default=False)

# -----------------------------
# Daily email digest — off by default. Flip EMAIL_DIGEST_ENABLED=true in .env
# once EMAIL_DIGEST_RECIPIENT and SMTP_* below hold real values.
# -----------------------------
EMAIL_DIGEST_ENABLED = _bool_env("EMAIL_DIGEST_ENABLED", default=False)
EMAIL_DIGEST_RECIPIENT = os.environ.get("EMAIL_DIGEST_RECIPIENT", "xxx@xxxx")  # placeholder, fill in .env
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM") or SMTP_USER


def validate_for_search() -> List[str]:
    """Human-readable problems that would block step 1 (search/download) from running."""
    problems = []
    if not ENTREZ_EMAIL:
        problems.append("ENTREZ_EMAIL is not set (required by NCBI Eutils usage policy).")
    if not UNPAYWALL_EMAIL:
        problems.append("UNPAYWALL_EMAIL is not set (required by the Unpaywall API).")
    return problems


def validate_for_institutional_proxy() -> List[str]:
    """Human-readable problems that would block the institutional-proxy fallback."""
    problems = []
    if "{url}" not in INSTITUTIONAL_PROXY_URL_TEMPLATE:
        problems.append("INSTITUTIONAL_PROXY_URL_TEMPLATE is unset or missing a '{url}' placeholder.")
    if not INSTITUTIONAL_PROXY_COOKIES_FILE.is_file():
        problems.append(f"cookies file not found: {INSTITUTIONAL_PROXY_COOKIES_FILE}")
    return problems


def validate_for_email() -> List[str]:
    """Human-readable problems that would block the email digest from sending."""
    problems = []
    if EMAIL_DIGEST_RECIPIENT in ("", "xxx@xxxx"):
        problems.append("EMAIL_DIGEST_RECIPIENT is still the placeholder value.")
    if not SMTP_HOST:
        problems.append("SMTP_HOST is not set.")
    if not SMTP_USER or not SMTP_PASSWORD:
        problems.append("SMTP_USER / SMTP_PASSWORD are not set.")
    return problems