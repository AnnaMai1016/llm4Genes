#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PaperReviewTool.src.gpt_extractor import PaperExtractor
from PaperReviewTool.src.patches import PatchSet
from PaperReviewTool.src.schema.meta_stru import ExtractionSchema
from PaperReviewTool.src.util import read_full_paper


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def setup_logger(global_log: Path, scenario_log: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("assessment.patch_validate_apply")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for log_path in [global_log, scenario_log]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def load_patch_set(parsed: dict[str, Any]) -> PatchSet:
    patch_block = parsed.get("patch_set")
    if isinstance(patch_block, dict):
        return PatchSet.model_validate(patch_block)
    if isinstance(parsed.get("patches"), list):
        return PatchSet.model_validate({"patches": parsed["patches"]})
    raise ValueError("No patch_set found in verification parsed output.")


def discover_markdown(sample_root: Path, paper_id: str) -> Path | None:
    folder = sample_root / "mineru" / paper_id / "vlm"
    if not folder.is_dir():
        return None
    preferred = folder / f"{paper_id}.md"
    if preferred.is_file():
        return preferred
    mds = sorted(folder.glob("*.md"))
    return mds[0] if mds else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially validate and apply verification patches.")
    parser.add_argument("--runs-dir", default="assessment/runs")
    parser.add_argument("--sample-root", default="assessment/samples")
    parser.add_argument("--verification-scenario", required=True)
    parser.add_argument("--base-scenario", required=True)
    parser.add_argument("--turn-index", type=int, default=1)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--verify-with-llm", action="store_true")
    parser.add_argument("--min-match-score", type=float, default=40.0)
    parser.add_argument("--context-window-padding", type=int, default=5)
    parser.add_argument("--context-words", type=int, default=50)
    parser.add_argument("--max-matches", type=int, default=3)
    parser.add_argument("--no-strict-old-value", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()

    runs_dir = REPO_ROOT / args.runs_dir
    sample_root = REPO_ROOT / args.sample_root
    verification_root = runs_dir / args.verification_scenario
    base_root = runs_dir / args.base_scenario

    if not verification_root.is_dir():
        raise FileNotFoundError(f"verification scenario not found: {verification_root}")
    if not base_root.is_dir():
        raise FileNotFoundError(f"base scenario not found: {base_root}")

    logger = setup_logger(
        REPO_ROOT / "assessment" / "00_log_cursor.log",
        verification_root / "logs" / "run.log",
        args.verbose,
    )
    logger.info(
        "patch validate/apply start | verification_scenario=%s | base_scenario=%s | turn=%d",
        args.verification_scenario,
        args.base_scenario,
        args.turn_index,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "DUMMY")
    client = OpenAI(api_key=api_key)

    verification_results = verification_root / "results"
    base_results = base_root / "results"
    turn_suffix = f"turn_{args.turn_index:02d}"

    paper_dirs = sorted([p for p in verification_results.glob("*") if p.is_dir()])
    if args.max_papers is not None:
        paper_dirs = paper_dirs[: args.max_papers]

    summary_rows: list[dict[str, Any]] = []
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        parsed_path = paper_dir / f"{turn_suffix}_parsed.json"
        base_merged_path = base_results / paper_id / "merged_extraction.json"
        if not parsed_path.is_file():
            summary_rows.append({"paper_id": paper_id, "status": "skipped", "reason": "missing_verification_parsed"})
            continue
        if not base_merged_path.is_file():
            summary_rows.append({"paper_id": paper_id, "status": "skipped", "reason": "missing_base_merged_extraction"})
            continue

        md_path = discover_markdown(sample_root, paper_id)
        if md_path is None:
            summary_rows.append({"paper_id": paper_id, "status": "skipped", "reason": "missing_markdown"})
            continue

        try:
            parsed = read_json(parsed_path)
            if not isinstance(parsed, dict):
                raise ValueError("verification parsed payload is not a dict")
            patch_set = load_patch_set(parsed)
            base_extraction = ExtractionSchema.model_validate(read_json(base_merged_path))

            extractor = PaperExtractor(
                schema=ExtractionSchema,
                client=client,
                model=args.model,
                file_unique_id=paper_id,
                log_dir=paper_dir / "patch_ops",
            )
            extractor.set_document(read_full_paper(str(md_path)))
            extractor.current_result = base_extraction
            extractor.unused_patch_set = patch_set

            extractor.verify_patches(
                min_match_score=args.min_match_score,
                context_window_padding=args.context_window_padding,
                context_words=args.context_words,
                max_matches=args.max_matches,
                verify_with_llm=args.verify_with_llm,
                clear_unused_patch_set=False,
            )

            verification_rows = [
                pv.model_dump(mode="python", exclude_none=False) for pv in extractor.patch_verification_results
            ]
            write_json(paper_dir / f"{turn_suffix}_patch_verification_results.json", verification_rows)

            extractor.apply_verified_patches(
                strict_old_value=not args.no_strict_old_value,
                clear_patch_verification_results=False,
            )

            applied_result = (
                extractor.current_result.model_dump(mode="python", exclude_none=False)
                if extractor.current_result is not None
                else None
            )
            write_json(paper_dir / f"{turn_suffix}_applied_extraction.json", applied_result)
            write_json(paper_dir / "merged_extraction.json", applied_result)

            verified_count = sum(1 for pv in extractor.patch_verification_results if pv.is_verified)
            summary_rows.append(
                {
                    "paper_id": paper_id,
                    "status": "ok",
                    "patches_total": len(extractor.patch_verification_results),
                    "patches_verified": verified_count,
                    "patches_rejected": len(extractor.patch_verification_results) - verified_count,
                }
            )
        except Exception as exc:
            logger.exception("patch validate/apply failed | paper_id=%s", paper_id)
            summary_rows.append({"paper_id": paper_id, "status": "failed", "error": str(exc)})

    summary = {
        "saved_at_utc": now_utc(),
        "verification_scenario": args.verification_scenario,
        "base_scenario": args.base_scenario,
        "turn_index": args.turn_index,
        "verify_with_llm": args.verify_with_llm,
        "strict_old_value": not args.no_strict_old_value,
        "papers_processed": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(verification_root / "summary" / f"{turn_suffix}_patch_apply_summary.json", summary)
    logger.info("patch validate/apply done | papers=%d", len(summary_rows))


if __name__ == "__main__":
    main()
