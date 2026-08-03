"""
Manual, single-paper fallback: fetch ONE specific paper through your own
institutional library proxy (e.g. EZproxy), for an article your school
already has a legitimate subscription to. This replays your own logged-in
browser session — it is not a paywall bypass.

READ THIS FIRST: your library's Electronic Resource Use Policy (UIUC's, and
most schools') explicitly prohibits "systematic downloading of electronic
resources using robots, spiders or manual means," and publishers cut off
proxied access for the ENTIRE INSTITUTION — not just your account — when they
detect that pattern. That's why this module is NOT wired into
`download_manager.resolve_fulltext_candidates` / `run()` / `run_daily.py`:
never call `download_manager.fetch_via_institutional_proxy` in a loop over
search results or on any kind of schedule. Use it by hand, one DOI at a time,
for a specific paper you are personally reading right now.

Off by default (`config.INSTITUTIONAL_PROXY_ENABLED`). One-time setup:

  1. Find your library's proxy login URL. Usually on your library's
     "off-campus access" / "EZproxy" page, or visible in the address bar when
     you click through to a paywalled article from an on-campus library
     search. It typically looks like `https://<proxy-host>/login?url=`.
     Set in `PaperSearch/.env`:
         INSTITUTIONAL_PROXY_URL_TEMPLATE=https://<proxy-host>/login?url={url}
  2. Log into that URL in your normal browser (once — the session usually
     lasts hours to a day, sometimes longer).
  3. Export your browser cookies for that domain to a Netscape-format
     cookies.txt file — e.g. the "Get cookies.txt LOCALLY" extension for
     Chrome/Firefox — and save it to the path in
     `config.INSTITUTIONAL_PROXY_COOKIES_FILE` (default:
     `PaperSearch/data/institutional_cookies.txt`).
  4. Set INSTITUTIONAL_PROXY_ENABLED=true in `.env`.

We never see or store your NetID/password — only the cookie file you export
yourself, which you re-export whenever the session expires (downloads through
this path will start failing again as the signal that it has).
"""

from __future__ import annotations

import http.cookiejar
import logging
from typing import Optional
from urllib.parse import quote

from .. import config

logger = logging.getLogger("papersearch.institutional_proxy")

_cookies_cache: Optional[dict[str, str]] = None


def available() -> bool:
    return not config.validate_for_institutional_proxy() and config.INSTITUTIONAL_PROXY_ENABLED


def proxied_url(target_url: str) -> str:
    return config.INSTITUTIONAL_PROXY_URL_TEMPLATE.format(url=quote(target_url, safe=""))


def load_cookies() -> Optional[dict[str, str]]:
    """Load the exported cookies.txt once per process; None if unreadable/absent."""
    global _cookies_cache
    if _cookies_cache is not None:
        return _cookies_cache
    path = config.INSTITUTIONAL_PROXY_COOKIES_FILE
    if not path.is_file():
        return None
    jar = http.cookiejar.MozillaCookieJar(str(path))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception:
        logger.exception("failed to parse cookies file %s", path)
        return None
    _cookies_cache = {c.name: c.value for c in jar}
    return _cookies_cache