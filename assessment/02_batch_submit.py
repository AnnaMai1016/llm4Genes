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

from PaperReviewTool.src.prompt import ExtractionPrompt, build_user_prompt
from PaperReviewTool.src.schema.meta_stru import ExtractionSchema
from PaperReviewTool.src.util import read_full_paper


DEFAULT_TURN_FIELDS: list[list[str]] = [
    ["metadata", "relevance"],
    ["context"],
    ["scale", "methodology", "tile_design"],
    ["results", "internal_evaluation"],
    ["key_words", "overall_synthesis"],
]

ALL_TOP_LEVEL_FIELDS: list[str] = [
    "metadata",
    "relevance",
    "key_words",
    "context",
    "scale",
    "methodology",
    "tile_design",
    "results",
    "internal_evaluation",
    "overall_synthesis",
]

VERIFICATION_FIELDS: list[str] = [
    "/relevance",
    "/location",
    "/context",
    "/scale",
    "/methodology",
    "/tile_design",
    "/results/summary_bullets",
    "/internal_evaluation",
]


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_json_list(path: Path, item: Any) -> None:
    rows = []
    if path.is_file():
        existing = read_json(path)
        if isinstance(existing, list):
            rows = existing
    rows.append(item)
    write_json(path, rows)


def build_scenario_name(model: str, sample_meta: str, scenario_name: str | None) -> str:
    if scenario_name:
        return scenario_name
    stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    sample_tag = Path(sample_meta).stem
    return f"extract__{model}__{sample_tag}__{stamp}"


def scenario_paths(runs_dir: Path, scenario_name: str) -> dict[str, Path]:
    root = runs_dir / scenario_name
    return {
        "root": root,
        "config": root / "config",
        "logs": root / "logs",
        "manifests": root / "manifests",
        "batch": root / "batch",
        "results": root / "results",
        "summary": root / "summary",
    }


def setup_logger(global_log: Path, scenario_log: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("assessment.batch_submit")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

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


def load_turn_fields(turns_config: Path | None) -> list[list[str]]:
    if turns_config is None:
        return DEFAULT_TURN_FIELDS
    data = read_json(turns_config)
    if not isinstance(data, list):
        raise ValueError("turns config must be list[list[str]]")
    out: list[list[str]] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, list) or not all(isinstance(v, str) for v in item):
            raise ValueError(f"Invalid turns config at position {i}")
        out.append(item)
    return out


def discover_markdown(sample_root: Path, paper_id: str) -> Path | None:
    folder = sample_root / "mineru" / paper_id / "vlm"
    if not folder.is_dir():
        return None
    preferred = folder / f"{paper_id}.md"
    if preferred.is_file():
        return preferred
    mds = sorted(folder.glob("*.md"))
    return mds[0] if mds else None


def load_previous_result(results_dir: Path, paper_id: str) -> dict[str, Any] | None:
    path = results_dir / paper_id / "merged_extraction.json"
    if not path.is_file():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def build_response_body(
    model: str,
    document_text: str,
    focus_area: list[str],
    previous_result: dict[str, Any] | None,
    strategy: str,
    prompt_cache_key: str,
    reasoning_effort: str,
    prompt_cache_retention: str = "24h",
) -> dict[str, Any]:
    # if reasoning_effort not in {"none", "low", "medium", "high"}:
    #     raise ValueError(f"invalid reasoning_effort: {reasoning_effort}, must be one of low, medium, high")
    if strategy in {"update", "verification"}:
        if previous_result is None:
            raise ValueError(f"strategy={strategy} requires previous_result")
        user_prompt = build_user_prompt(
            full_text=document_text,
            focus_area=focus_area,
            updation=(strategy == "update"),
            verification=(strategy == "verification"),
            previous_extraction=ExtractionSchema.model_validate(previous_result),
        )
    # elif previous_result is None:
    #     user_prompt = build_user_prompt(full_text=document_text, focus_area=focus_area)
    # else:
    #     user_prompt = build_user_prompt(
    #         full_text=document_text,
    #         focus_area=focus_area,
    #         updation=True,
    #         previous_extraction=ExtractionSchema.model_validate(previous_result),
    #     )
    else: 
        user_prompt = build_user_prompt(full_text=document_text,focus_area=focus_area)

    text_format = {
        "type": "json_schema",
        "name": "ExtractionSchema",
        "schema": ExtractionSchema.model_json_schema(),
        "strict": False,
    }

    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": ExtractionPrompt.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        # Keep the payload close to PaperExtractor._response():
        # - text verbosity lives under `text`
        # - schema is supplied via `text_format`
        "text": {
            "verbosity": "medium",
            "format": text_format,
        },
        "prompt_cache_retention": prompt_cache_retention,
        "prompt_cache_key": prompt_cache_key,
        "reasoning": {"effort": reasoning_effort},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and submit one turn of OpenAI batch extraction requests."
    )
    parser.add_argument("--sample-meta", default="assessment/samples/sample_meta.json")
    parser.add_argument("--sample-root", default="assessment/samples")
    parser.add_argument("--runs-dir", default="assessment/runs")
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--turns-config", default=None)
    parser.add_argument("--turn-index", type=int, required=True, help="1-based turn index to submit")
    parser.add_argument(
        "--strategy",
        choices=["single-path", "multi-path", "update", "verification"],
        default="multi-path",
        help="single-path: all fields in one run; multi-path: split by turns; update/verification: from previous extraction.",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--base-scenario",
        default=None,
        help="When set, read previous merged outputs from assessment/runs/<base-scenario>/results/",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prompt-cache-retention", default="24h")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="medium")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()

    sample_meta = REPO_ROOT / args.sample_meta
    sample_root = REPO_ROOT / args.sample_root
    runs_dir = REPO_ROOT / args.runs_dir
    turns_config = REPO_ROOT / args.turns_config if args.turns_config else None

    turn_fields = load_turn_fields(turns_config)
    has_turns_config = turns_config is not None

    if args.strategy == "single-path":
        if args.turn_index != 1:
            raise ValueError("--strategy single-path requires --turn-index 1")
        focus_area: list[str] | None = None
    elif args.strategy == "multi-path":
        if args.turn_index < 1 or args.turn_index > len(turn_fields):
            raise ValueError(f"--turn-index must be in [1, {len(turn_fields)}]")
        focus_area = turn_fields[args.turn_index - 1]
    elif args.strategy == "update":
        if has_turns_config:
            if args.turn_index < 1 or args.turn_index > len(turn_fields):
                raise ValueError(f"--turn-index must be in [1, {len(turn_fields)}]")
            focus_area = turn_fields[args.turn_index - 1]
        else:
            if args.turn_index != 1:
                raise ValueError("--strategy update without --turns-config requires --turn-index 1")
            # Default update scope is all fields in one run.
            # focus_area = ALL_TOP_LEVEL_FIELDS
            focus_area = None
    else:  # verification
        if has_turns_config:
            if args.turn_index < 1 or args.turn_index > len(turn_fields):
                raise ValueError(f"--turn-index must be in [1, {len(turn_fields)}]")
            focus_area = turn_fields[args.turn_index - 1]
        else:
            if args.turn_index != 1:
                raise ValueError("--strategy verification without --turns-config requires --turn-index 1")
            # Default verification scope is all fields in one run.
            # focus_area = ALL_TOP_LEVEL_FIELDS
            focus_area = VERIFICATION_FIELDS

    scenario = build_scenario_name(args.model, args.sample_meta, args.scenario_name)
    paths = scenario_paths(runs_dir, scenario)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        REPO_ROOT / "assessment" / "00_log_cursor.log",
        paths["logs"] / "run.log",
        args.verbose,
    )
    logger.info(
        "submission started | scenario=%s | strategy=%s | turn=%d",
        scenario,
        args.strategy,
        args.turn_index,
    )

    base_results_dir = paths["results"]
    if args.base_scenario:
        base_results_dir = runs_dir / args.base_scenario / "results"
        if not base_results_dir.is_dir():
            raise FileNotFoundError(f"base scenario results folder not found: {base_results_dir}")

    submit_config = {
        "saved_at_utc": now_utc(),
        "scenario_name": scenario,
        "strategy": args.strategy,
        "turn_index": args.turn_index,
        "focus_area": focus_area,
        "model": args.model,
        "sample_meta": str(sample_meta),
        "sample_root": str(sample_root),
        "max_papers": args.max_papers,
        "turns_config": str(turns_config) if turns_config else None,
        "base_scenario": args.base_scenario,
        "base_results_dir": str(base_results_dir),
        "dry_run": args.dry_run,
        "reasoning_effort": args.reasoning_effort,
        "prompt_cache_retention": args.prompt_cache_retention,
    }
    # Keep one immutable snapshot per turn to avoid overwriting
    # when running sequential turns in the same scenario.
    write_json(paths["config"] / f"submit_config_turn_{args.turn_index:02d}.json", submit_config)

    rows = read_json(sample_meta)
    if not isinstance(rows, list):
        raise ValueError("sample meta must be a JSON array")
    if args.max_papers is not None:
        rows = rows[: args.max_papers]

    turn_dir = paths["batch"] / f"turn_{args.turn_index:02d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    request_jsonl = turn_dir / "requests.jsonl"

    missing_inputs: list[dict[str, Any]] = []
    request_count = 0
    with request_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            doi = str(row.get("doi", "")).strip()
            if not doi:
                continue
            paper_id = sanitize_filename(doi)
            md_path = discover_markdown(sample_root, paper_id)
            if md_path is None:
                missing_inputs.append({"paper_id": paper_id, "doi": doi, "reason": "missing_markdown"})
                continue

            # Multi-path extraction turns are independent; only update/verification
            # require loading previous extraction results.
            needs_previous = args.strategy in {"update", "verification"}
            previous_result = load_previous_result(base_results_dir, paper_id) if needs_previous else None
            if needs_previous and previous_result is None:
                missing_inputs.append(
                    {"paper_id": paper_id, "doi": doi, "reason": "missing_previous_merged_extraction"}
                )
                continue

            body = build_response_body(
                model=args.model,
                # Reuse existing utility to trim non-content sections
                # (e.g., references/acknowledgements) before prompting.
                document_text=read_full_paper(str(md_path)),
                focus_area=focus_area,
                previous_result=previous_result,
                strategy=args.strategy,
                prompt_cache_key=doi,
                prompt_cache_retention=args.prompt_cache_retention,
                reasoning_effort=args.reasoning_effort,
            )
            custom_id = f"{paper_id}__{args.strategy}__turn_{args.turn_index:02d}"
            request_row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            out.write(json.dumps(request_row, ensure_ascii=False) + "\n")
            request_count += 1

    write_json(turn_dir / "missing_inputs.json", missing_inputs)
    logger.info("requests built | count=%d | missing=%d", request_count, len(missing_inputs))
    if request_count == 0:
        logger.warning("No requests created for this turn. Exiting.")
        return

    if args.dry_run:
        logger.info("dry-run enabled. Batch submission skipped.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    with request_jsonl.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "scenario_name": scenario,
            "strategy": args.strategy,
            "turn_index": str(args.turn_index),
            "focus_area": ",".join(focus_area) if focus_area else "__ALL_FIELDS__",
            "model": args.model,
        },
    )

    metadata = {
        "saved_at_utc": now_utc(),
        "scenario_name": scenario,
        "strategy": args.strategy,
        "turn_index": args.turn_index,
        "focus_area": focus_area,
        "model": args.model,
        "request_count": request_count,
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "batch_status": getattr(batch, "status", None),
        "request_jsonl": str(request_jsonl),
        "base_scenario": args.base_scenario,
        "base_results_dir": str(base_results_dir),
    }
    write_json(turn_dir / "batch_meta.json", metadata)
    append_json_list(paths["manifests"] / "submit_manifest.json", metadata)
    logger.info("batch submitted | batch_id=%s | input_file_id=%s", batch.id, uploaded.id)


if __name__ == "__main__":
    main()
