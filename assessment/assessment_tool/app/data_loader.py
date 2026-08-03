"""Data discovery and loading helpers for the assessment tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CASE_MAP = {
    "s1_gpt54_single": "GPT-5.4-mini single-path baseline",
    "s2_gpt5_single": "GPT-5-mini single-path baseline",
    "s3_gpt5_multi": "GPT-5-mini multi-path extraction",
    "s4_gpt5_update1": "GPT-5-mini extraction + 1-turn update",
    "s5_gpt5_update2": "GPT-5-mini extraction + 2-turn update",
    "s6_gpt5_verify": "GPT-5-mini extraction + verification",
    "s7_gpt5_update_verify": "GPT-5-mini update + verification",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def markdown_root() -> Path:
    return project_root() / "assessment" / "samples" / "mineru"


def runs_root() -> Path:
    return project_root() / "assessment" / "runs"


def list_cases() -> list[str]:
    return [case for case in CASE_MAP if (runs_root() / case / "results").exists()]


def list_paper_ids() -> list[str]:
    papers: set[str] = set()
    for md in markdown_root().glob("*/vlm/*.md"):
        papers.add(md.parent.parent.name)
    return sorted(papers)


def paper_md_path(paper_id: str) -> Path:
    """MinerU output: `{paper_id}.md` alongside `{paper_id}_origin.pdf` in `vlm/`."""
    return markdown_root() / paper_id / "vlm" / f"{paper_id}.md"


def paper_pdf_origin_path(paper_id: str) -> Path:
    """Original PDF path paired with `{paper_id}.md` (naming: `xx_origin.pdf` ↔ `xx.md`)."""
    return markdown_root() / paper_id / "vlm" / f"{paper_id}_origin.pdf"


def paper_markdown_path(paper_id: str) -> Path:
    """Deprecated alias for `paper_md_path` (same file)."""
    return paper_md_path(paper_id)


def extraction_path(case_id: str, paper_id: str) -> Path:
    return runs_root() / case_id / "results" / paper_id / "merged_extraction.json"


def load_md(paper_id: str) -> str:
    path = paper_md_path(paper_id)
    if not path.exists():
        raise FileNotFoundError(f".md file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_markdown(paper_id: str) -> str:
    """Deprecated alias for `load_md`."""
    return load_md(paper_id)


def load_extraction(case_id: str, paper_id: str) -> dict[str, Any]:
    path = extraction_path(case_id, paper_id)
    if not path.exists():
        raise FileNotFoundError(f"Extraction not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_fields(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested JSON into field paths and leaf values."""
    out: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            out.extend(flatten_fields(value, next_prefix))
        return out
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            next_prefix = f"{prefix}[{idx}]"
            out.extend(flatten_fields(value, next_prefix))
        if not payload:
            out.append((prefix, []))
        return out
    out.append((prefix, payload))
    return out

