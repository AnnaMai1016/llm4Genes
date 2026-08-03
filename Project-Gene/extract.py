#!/usr/bin/env python3
"""
Step 3: run the SWEET entity/relation extraction (PaperReviewTool's
`PaperExtractor` + `schema/sweet_transporter.py`) over every paper whose
Markdown is ready, and record the result.

For each manifest row with `mineru_status="done"` (and not yet extracted):
  1. Clean the Markdown with PaperReviewTool's `read_full_paper()` — strips
     References/Acknowledgements/etc. before it reaches the LLM.
  2. Run one `extract_turn()` per paper — one API call per
     `sweet_transporter.NAME_FIELDS` group (metadata, entities, relations),
     merged into a single `ExtractionSchema` result.
  3. Write the result to `<paper_id>/extraction.json` (next to MinerU's own
     `<paper_id>/auto/` output) and record `extraction_status` /
     `extraction_path` / `entities_count` / `relations_count` back onto the
     same manifest row.

Unlike step 2 (MinerU), this step only makes OpenAI API calls — no local
model server, no thread explosion — so it's fine to run directly on the
login node, same as PaperSearch's scripts.

Usage (from repo root, `llmReview` conda env active):
    python Project-Gene/extract.py [--max-papers N] [--model gpt-5.4-mini]

Idempotent: manifest rows already marked extraction_status="done" are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

import config as gene_config  # this directory's own config.py
from PaperSearch.manifest import load_manifest, save_manifest
from PaperSearch.utils import sanitize_filename
from PaperReviewTool.src.gpt_extractor import PaperExtractor
from PaperReviewTool.src.schema.sweet_transporter import EXTRACTION_PROMPT, NAME_FIELDS, ExtractionSchema
from PaperReviewTool.src.util import read_full_paper

logger = logging.getLogger("project_gene.extract")


def extract_one(client: OpenAI, markdown_path: Path, paper_id: str, model: str) -> Optional[dict[str, Any]]:
    document_text = read_full_paper(str(markdown_path))
    if not document_text.strip():
        logger.error("read_full_paper produced empty text for %s", markdown_path)
        return None

    extractor = PaperExtractor(
        schema=ExtractionSchema,
        prompt=EXTRACTION_PROMPT,
        client=client,
        model=model,
        file_unique_id=paper_id,
        log_dir=gene_config.EXTRACTION_LOG_DIR,
    )
    extractor.set_document(document_text)
    extractor.extract_turn(NAME_FIELDS=NAME_FIELDS)

    if extractor.current_result is None:
        return None
    return extractor.current_result.model_dump(mode="python", exclude_none=True)


def run(max_papers: Optional[int] = None, model: Optional[str] = None) -> list[dict[str, Any]]:
    problems = gene_config.validate_for_extraction()
    if problems:
        raise RuntimeError("Project-Gene config incomplete: " + "; ".join(problems))
    model = model or gene_config.EXTRACTION_MODEL
    client = OpenAI(api_key=gene_config.OPENAI_API_KEY)

    rows = load_manifest(gene_config.MANIFEST_PATH)
    pending = [
        r for r in rows
        if r.get("mineru_status") == "done" and r.get("extraction_status") != "done"
    ]
    if max_papers is not None:
        pending = pending[:max_papers]

    touched: list[dict[str, Any]] = []
    for row in pending:
        doi = row["doi"]
        paper_id = sanitize_filename(doi)
        markdown_path = Path(row["markdown_path"]) if row.get("markdown_path") else None
        if not markdown_path or not markdown_path.is_file():
            row["extraction_status"] = "missing_markdown"
            touched.append(row)
            save_manifest(gene_config.MANIFEST_PATH, rows)
            continue

        try:
            result = extract_one(client, markdown_path, paper_id, model)
        except Exception:
            logger.exception("extraction failed for doi=%s", doi)
            result = None

        if result is None:
            row["extraction_status"] = "failed"
            touched.append(row)
            save_manifest(gene_config.MANIFEST_PATH, rows)
            continue

        out_path = gene_config.OUTPUT_DIR / paper_id / "extraction.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        row["extraction_status"] = "done"
        row["extraction_path"] = str(out_path)
        row["entities_count"] = len(result.get("entities") or [])
        row["relations_count"] = len(result.get("relations") or [])
        logger.info(
            "extracted | doi=%s | entities=%d | relations=%d -> %s",
            doi, row["entities_count"], row["relations_count"], out_path,
        )
        touched.append(row)
        save_manifest(gene_config.MANIFEST_PATH, rows)

    return touched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract SWEET entities/relations from converted Markdown.")
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--model", default=None, help="Override EXTRACTION_MODEL from .env for this run.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    done = run(max_papers=args.max_papers, model=args.model)
    print(f"processed {len(done)} paper(s)")
