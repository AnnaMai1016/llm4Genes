"""
Optional daily email digest, gated by the `EMAIL_DIGEST_ENABLED` switch in
`config.py` / `.env`. Off by default; `send_daily_digest` is a no-op until you
set `EMAIL_DIGEST_ENABLED=true` and fill in a real `EMAIL_DIGEST_RECIPIENT`
and `SMTP_*` credentials.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from typing import Any

from . import config

logger = logging.getLogger("papersearch.notifier")


def _format_digest(new_rows: list[dict[str, Any]]) -> str:
    if not new_rows:
        return "PaperSearch: no new papers found or converted today."
    lines = [f"PaperSearch daily digest — {len(new_rows)} paper(s) touched today:", ""]
    for row in new_rows:
        lines.append(f"- {row.get('title') or row.get('doi')}")
        lines.append(
            f"  doi={row.get('doi')}  fulltext={row.get('fulltext_status', 'n/a')}"
            f"  mineru={row.get('mineru_status', 'pending')}"
        )
    return "\n".join(lines)


def send_daily_digest(new_rows: list[dict[str, Any]]) -> bool:
    """Send the digest if the switch is on. Returns True iff an email was actually sent."""
    if not config.EMAIL_DIGEST_ENABLED:
        logger.info("email digest disabled (EMAIL_DIGEST_ENABLED=false); skipping")
        return False

    problems = config.validate_for_email()
    if problems:
        logger.error("email digest enabled but misconfigured: %s", "; ".join(problems))
        return False

    body = _format_digest(new_rows)
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"PaperSearch daily digest ({len(new_rows)} paper(s))"
    msg["From"] = config.SMTP_FROM
    msg["To"] = config.EMAIL_DIGEST_RECIPIENT

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("digest sent to %s", config.EMAIL_DIGEST_RECIPIENT)
        return True
    except Exception:
        logger.exception("failed to send email digest")
        return False