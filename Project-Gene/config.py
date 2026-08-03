"""
Configuration for Project-Gene: step 2 (MinerU PDF -> Markdown conversion)
and step 3 (LLM entity/relation extraction over that Markdown).

Reuses PaperSearch's manifest.json as the shared source of truth for which
papers exist (DOI -> pdf_path -> markdown_path -> extraction_path); each step
writes back its own status fields onto the same rows so the whole pipeline's
state lives in one place.

Copy `.env.example` to `.env` in this same directory to override defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)

# Shared state from step 1 (PaperSearch).
MANIFEST_PATH = Path(
    os.environ.get("MANIFEST_PATH") or (REPO_ROOT / "PaperSearch" / "data" / "manifest.json")
)

# -----------------------------
# Step 2: MinerU conversion
# -----------------------------
# Converted Markdown (and MinerU's other output files: images/, *_middle.json,
# *_content_list.json, ...) land in this directory itself, one subfolder per paper.
OUTPUT_DIR = PROJECT_ROOT

MINERU_BIN = os.environ.get("MINERU_BIN", "mineru")
# "pipeline": CPU-only, no GPU needed — this cluster's login node has none,
# and getting a GPU allocation for vlm-engine/hybrid-engine (much higher
# parsing quality, but needs torch+vllm and a Slurm GPU partition) is a
# bigger step for later. Switch via MINERU_BACKEND once that's set up.
MINERU_BACKEND = os.environ.get("MINERU_BACKEND", "pipeline")
MINERU_LANG = os.environ.get("MINERU_LANG", "en")  # OCR language hint; pipeline/hybrid backends only

# -----------------------------
# Step 3: LLM extraction (PaperReviewTool + the SWEET schema)
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Matches PaperExtractor's own default; override here rather than editing gpt_extractor.py.
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-5.4-mini")
EXTRACTION_LOG_DIR = Path(os.environ.get("EXTRACTION_LOG_DIR") or (PROJECT_ROOT / "logs"))


# -----------------------------
# Step 4: entity resolution + graph aggregation
# -----------------------------
# Canonical-entity table (name/synonym -> canonical id), persisted across runs.
ENTITY_CANON_PATH = Path(os.environ.get("ENTITY_CANON_PATH") or (PROJECT_ROOT / "data" / "entity_canon.json"))
# Assembled graph output (nodes + edges), rebuilt each run from the canon table
# + every paper's extraction.json.
GRAPH_OUTPUT_DIR = Path(os.environ.get("GRAPH_OUTPUT_DIR") or (PROJECT_ROOT / "graph"))
# difflib.get_close_matches cutoff (0-1) for snapping a new entity name onto an
# existing canonical one when no exact/alias match is found. Higher = stricter.
ENTITY_FUZZY_CUTOFF = float(os.environ.get("ENTITY_FUZZY_CUTOFF", "0.85"))


def validate_for_mineru() -> List[str]:
    problems = []
    if not MANIFEST_PATH.is_file():
        problems.append(f"manifest not found: {MANIFEST_PATH} (run PaperSearch step 1 first)")
    return problems


def validate_for_extraction() -> List[str]:
    problems = []
    if not MANIFEST_PATH.is_file():
        problems.append(f"manifest not found: {MANIFEST_PATH} (run PaperSearch step 1 first)")
    if not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY is not set in Project-Gene/.env")
    return problems
