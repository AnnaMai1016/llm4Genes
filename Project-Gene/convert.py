#!/usr/bin/env python3
"""
Step 2: convert every downloaded PDF (tracked in PaperSearch's manifest) into
Markdown via MinerU, storing output in this directory.

MUST be run inside a Slurm allocation — never directly on the login node.
MinerU's pipeline backend starts a local model-serving subprocess that spawns
per-visible-core worker threads (onnxruntime/torch default to one thread per
CPU the machine *reports*, 128 here, regardless of how many a job actually
gets). Run raw on the login node, that has already once exhausted this
account's shared process/thread quota badly enough to break `fork()` for
everyone using it, and — unlike under Slurm — nothing reliably tears the
whole process tree down if a `timeout` wrapper doesn't catch every child.
Slurm's cgroup cleanup on job exit is what actually contains this.

Also: don't point OUTPUT_DIR (or anything else this script writes to) at
`/tmp` inside an srun job — many HPC setups give each Slurm job its own
private `/tmp` that's torn down with the job, so files written there vanish
the moment the job ends even though the write itself appeared to succeed.
Write to the real filesystem (this directory, on Lustre) instead, which is
what OUTPUT_DIR already defaults to.

Usage (from repo root, `llmReview` conda env active, `mineru` installed):
    srun --partition=cpu-interactive --account=<your-account> \\
         --cpus-per-task=8 --mem=16G --time=00:30:00 \\
         python Project-Gene/convert.py [--max-papers N] [--backend pipeline]

Idempotent: manifest rows already marked mineru_status="done" are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as gene_config  # this directory's own config.py (script dir is auto-added to sys.path)
from PaperSearch.manifest import load_manifest, save_manifest
from PaperSearch.utils import sanitize_filename

logger = logging.getLogger("project_gene.convert")

# See module docstring: cap thread pools before mineru (and the libraries it
# imports) get a chance to size themselves off the visible CPU count instead
# of the cgroup's actual allocation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _find_markdown(out_dir: Path, paper_id: str) -> Optional[Path]:
    """MinerU nests output as `<paper_id>/<method>/<paper_id>.md`, where
    `<method>` depends on backend/parse-method (e.g. "auto" for pipeline's
    default `-m auto`) and has changed across MinerU versions — don't
    hardcode it, just look for any single-level subfolder containing the
    expected filename (or failing that, any .md at all)."""
    paper_dir = out_dir / paper_id
    if not paper_dir.is_dir():
        return None
    preferred = sorted(paper_dir.glob(f"*/{paper_id}.md"))
    if preferred:
        return preferred[0]
    any_md = sorted(paper_dir.glob("*/*.md"))
    return any_md[0] if any_md else None


def convert_one(pdf_path: Path, paper_id: str, backend: str, lang: str) -> Optional[Path]:
    gene_config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [gene_config.MINERU_BIN, "-p", str(pdf_path), "-o", str(gene_config.OUTPUT_DIR), "-b", backend]
    if backend == "pipeline" and lang:
        cmd += ["-l", lang]
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("mineru failed for %s (exit %d): %s", pdf_path, result.returncode, result.stderr[-2000:])
        return None
    md_path = _find_markdown(gene_config.OUTPUT_DIR, paper_id)
    if md_path is None:
        logger.error(
            "mineru exited 0 for %s but no .md was found under %s — inspect it manually",
            pdf_path, gene_config.OUTPUT_DIR / paper_id,
        )
    return md_path


def run(max_papers: Optional[int] = None, backend: Optional[str] = None) -> list[dict[str, Any]]:
    problems = gene_config.validate_for_mineru()
    if problems:
        raise RuntimeError("Project-Gene config incomplete: " + "; ".join(problems))
    backend = backend or gene_config.MINERU_BACKEND

    rows = load_manifest(gene_config.MANIFEST_PATH)
    pending = [r for r in rows if r.get("pdf_path") and r.get("mineru_status") != "done"]
    if max_papers is not None:
        pending = pending[:max_papers]

    touched: list[dict[str, Any]] = []
    for row in pending:
        doi = row["doi"]
        paper_id = sanitize_filename(doi)
        pdf_path = Path(row["pdf_path"])
        if not pdf_path.is_file():
            row["mineru_status"] = "missing_pdf"
            touched.append(row)
            save_manifest(gene_config.MANIFEST_PATH, rows)
            continue

        md_path = convert_one(pdf_path, paper_id, backend, gene_config.MINERU_LANG)
        if md_path is not None:
            row["mineru_status"] = "done"
            row["markdown_path"] = str(md_path)
            row["mineru_backend"] = backend
            logger.info("converted | doi=%s -> %s", doi, md_path)
        else:
            row["mineru_status"] = "failed"
        touched.append(row)
        save_manifest(gene_config.MANIFEST_PATH, rows)

    return touched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert downloaded PDFs to Markdown via MinerU.")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--backend", default=None, help="Override MINERU_BACKEND from .env for this run.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if "SLURM_JOB_ID" not in os.environ:
        raise SystemExit(
            "Refusing to run on the login node directly — MinerU's local model server "
            "has already once exhausted this account's shared process/thread quota when "
            "run bare here. Submit via Slurm instead, e.g.:\n\n"
            "  srun --partition=cpu-interactive --account=<your-account> "
            "--cpus-per-task=8 --mem=16G --time=00:30:00 python Project-Gene/convert.py"
        )
    args = parse_args()
    done = run(max_papers=args.max_papers, backend=args.backend)
    print(f"processed {len(done)} paper(s); output under {gene_config.OUTPUT_DIR}")