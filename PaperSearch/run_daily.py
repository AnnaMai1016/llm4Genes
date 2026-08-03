#!/usr/bin/env python3
"""
Daily driver for step 1 only (search + download + email digest). Safe to run
repeatedly — idempotent, skips DOIs already downloaded via the manifest.

Step 2 (MinerU conversion, `Project-Gene/convert.py`) is intentionally NOT
called from here: it must run inside a Slurm allocation, not on the login
node this script runs on (see Project-Gene/convert.py's module docstring —
running it bare here has already once exhausted this account's shared
process/thread quota). Run it as its own, separately-submitted step.

Not scheduled automatically. Note: `crontab` is disabled on this account's
login node ("not allowed to access to (crontab) because of pam configuration")
— a cron entry will NOT work here. If you want this to run unattended, submit
it as a recurring Slurm job (e.g. a self-resubmitting `sbatch` script) or run
it from a machine that isn't a shared HPC login node.

Run manually with:  python -m PaperSearch.run_daily   (from the repo root)
"""

from __future__ import annotations

import logging

from . import download_manager, notifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("papersearch.run_daily")


def main() -> None:
    logger.info("=== PaperSearch daily run start ===")
    downloaded = download_manager.run()

    notifier.send_daily_digest(downloaded)
    logger.info(
        "=== PaperSearch daily run done | searched/downloaded=%d ===",
        len(downloaded),
    )


if __name__ == "__main__":
    main()