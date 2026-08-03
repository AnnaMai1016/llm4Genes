#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sdk_to_jsonable(obj: Any) -> Any:
    """Convert OpenAI SDK / pydantic objects to JSON-serializable values."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: sdk_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sdk_to_jsonable(x) for x in obj]
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return sdk_to_jsonable(model_dump())
    model_dict = getattr(obj, "dict", None)
    if callable(model_dict):
        return sdk_to_jsonable(model_dict())
    return str(obj)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def append_json_list(path: Path, item: Any) -> None:
    rows = []
    if path.is_file():
        existing = read_json(path)
        if isinstance(existing, list):
            rows = existing
    rows.append(item)
    write_json(path, rows)


def setup_logger(global_log: Path, scenario_log: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("assessment.batch_download")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    for log_path in [global_log, scenario_log]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def parse_custom_id(custom_id: str) -> tuple[str, str, int]:
    # New format: <paper_id>__<strategy>__turn_##.
    match_new = re.match(
        r"^(?P<paper>.+)__(?P<strategy>single-path|multi-path|update|verification)__turn_(?P<turn>\d+)$",
        custom_id,
    )
    if match_new:
        return match_new.group("paper"), match_new.group("strategy"), int(match_new.group("turn"))

    # Backward-compatible format: <paper_id>__turn_##.
    match_old = re.match(r"^(?P<paper>.+)__turn_(?P<turn>\d+)$", custom_id)
    if match_old:
        return match_old.group("paper"), "multi-path", int(match_old.group("turn"))

    raise ValueError(f"Invalid custom_id format: {custom_id}")


def deep_merge(base: Any, update: Any) -> Any:
    if update is None:
        return base
    if base is None:
        return update
    if isinstance(base, dict) and isinstance(update, dict):
        merged = dict(base)
        for k, v in update.items():
            merged[k] = deep_merge(merged.get(k), v)
        return merged
    return update


def extract_parsed_from_response_body(body: dict[str, Any]) -> dict[str, Any] | None:
    parsed = body.get("output_parsed")
    if isinstance(parsed, dict):
        return parsed

    output_text = body.get("output_text")
    if isinstance(output_text, str):
        text = output_text.strip()
        if text:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    output = body.get("output")
    if isinstance(output, list):
        for out_item in output:
            if not isinstance(out_item, dict):
                continue
            content = out_item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                txt = c.get("text")
                if isinstance(txt, str):
                    text = txt.strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        continue
    return None


def read_bytes_from_file_content(content_obj: Any) -> bytes:
    if hasattr(content_obj, "read"):
        data = content_obj.read()
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
    if hasattr(content_obj, "content"):
        data = content_obj.content
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
    if isinstance(content_obj, bytes):
        return content_obj
    if isinstance(content_obj, str):
        return content_obj.encode("utf-8")
    raise TypeError("Unknown file content response type")


def collect_jobs_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")
    jobs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("batch_id"):
            jobs.append(row)
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll and download OpenAI Batch API outputs.")
    parser.add_argument("--runs-dir", default="assessment/runs")
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--manifest", default=None, help="Default: <scenario>/manifests/submit_manifest.json")
    parser.add_argument("--batch-id", action="append", default=None, help="Optional explicit batch ids")
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--timeout-min", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()
    runs_dir = REPO_ROOT / args.runs_dir
    scenario_root = runs_dir / args.scenario_name
    scenario_log = scenario_root / "logs" / "run.log"
    logger = setup_logger(REPO_ROOT / "assessment" / "00_log_cursor.log", scenario_log, args.verbose)

    if not scenario_root.is_dir():
        raise FileNotFoundError(f"Scenario folder not found: {scenario_root}")

    manifest = Path(args.manifest) if args.manifest else scenario_root / "manifests" / "submit_manifest.json"
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest

    jobs = collect_jobs_from_manifest(manifest) if manifest.is_file() else []
    if args.batch_id:
        for batch_id in args.batch_id:
            jobs.append({"batch_id": batch_id, "scenario_name": args.scenario_name})

    if not jobs:
        raise ValueError("No jobs found. Provide --batch-id or a valid manifest.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    logger.info("download start | scenario=%s | jobs=%d", args.scenario_name, len(jobs))
    timeout_s = args.timeout_min * 60

    summary: dict[str, Any] = {"saved_at_utc": now_utc(), "scenario_name": args.scenario_name, "jobs": []}
    for job in jobs:
        batch_id = str(job["batch_id"])
        turn_index = int(job.get("turn_index", 0)) if str(job.get("turn_index", "")).isdigit() else None
        logger.info("polling batch | batch_id=%s", batch_id)

        batch_obj = None
        start = time.time()
        while True:
            batch_obj = client.batches.retrieve(batch_id)
            status = getattr(batch_obj, "status", None)
            snapshot = {
                "checked_at_utc": now_utc(),
                "batch_id": batch_id,
                "status": status,
                "output_file_id": getattr(batch_obj, "output_file_id", None),
                "error_file_id": getattr(batch_obj, "error_file_id", None),
                "request_counts": sdk_to_jsonable(
                    getattr(batch_obj, "request_counts", None)
                ),
                "usage": sdk_to_jsonable(
                    getattr(batch_obj, "usage", None)
                ),
            }
            append_json_list(scenario_root / "summary" / "batch_status_history.json", snapshot)
            logger.info("batch status | batch_id=%s | status=%s", batch_id, status)
            if status in TERMINAL_STATUSES:
                break
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            time.sleep(args.poll_interval_sec)

        status = getattr(batch_obj, "status", None)
        job_summary: dict[str, Any] = {
            "batch_id": batch_id,
            "status": status,
            "turn_index": turn_index,
            "output_file_id": getattr(batch_obj, "output_file_id", None),
            "error_file_id": getattr(batch_obj, "error_file_id", None),
            "request_counts": sdk_to_jsonable(
                getattr(batch_obj, "request_counts", None)
            ),
            "usage": sdk_to_jsonable(
                getattr(batch_obj, "usage", None)
            ),
        }

        if turn_index is None:
            turn_dir = scenario_root / "batch" / f"batch_{batch_id}"
        else:
            turn_dir = scenario_root / "batch" / f"turn_{turn_index:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        if status != "completed":
            write_json(turn_dir / "batch_status.json", job_summary)
            summary["jobs"].append(job_summary)
            continue

        output_file_id = getattr(batch_obj, "output_file_id", None)
        error_file_id = getattr(batch_obj, "error_file_id", None)
        output_path = turn_dir / "output.jsonl"
        error_path = turn_dir / "error.jsonl"

        if output_file_id and (args.force or not output_path.is_file()):
            content = client.files.content(output_file_id)
            output_path.write_bytes(read_bytes_from_file_content(content))
            logger.info("downloaded output file | batch_id=%s", batch_id)
        if error_file_id and (args.force or not error_path.is_file()):
            content = client.files.content(error_file_id)
            error_path.write_bytes(read_bytes_from_file_content(content))
            logger.info("downloaded error file | batch_id=%s", batch_id)

        parsed_dir = turn_dir / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        parsed_count = 0
        failed_count = 0
        if output_path.is_file():
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    custom_id = row.get("custom_id")
                    if not isinstance(custom_id, str):
                        continue
                    write_json(parsed_dir / f"{custom_id}.json", row)

                    try:
                        paper_id, strategy, custom_turn = parse_custom_id(custom_id)
                    except ValueError:
                        failed_count += 1
                        continue
                    response = row.get("response", {})
                    body = response.get("body", {}) if isinstance(response, dict) else {}
                    if not isinstance(body, dict):
                        body = {}
                    parsed = extract_parsed_from_response_body(body)
                    if parsed is None:
                        failed_count += 1
                        continue

                    results_paper_dir = scenario_root / "results" / paper_id
                    results_paper_dir.mkdir(parents=True, exist_ok=True)
                    write_json(results_paper_dir / f"turn_{custom_turn:02d}_raw.json", row)
                    write_json(results_paper_dir / f"turn_{custom_turn:02d}_parsed.json", parsed)

                    # Verification strategy returns patch suggestions and should not
                    # auto-merge into extraction results.
                    if strategy == "verification":
                        write_json(results_paper_dir / f"turn_{custom_turn:02d}_patch_response.json", parsed)
                    else:
                        merged_path = results_paper_dir / "merged_extraction.json"
                        merged = read_json(merged_path) if merged_path.is_file() else {}
                        if not isinstance(merged, dict):
                            merged = {}
                        merged = deep_merge(merged, parsed)
                        write_json(merged_path, merged)
                    parsed_count += 1

        job_summary.update(
            {
                "parsed_count": parsed_count,
                "parse_failed_count": failed_count,
                "output_path": str(output_path) if output_path.is_file() else None,
                "error_path": str(error_path) if error_path.is_file() else None,
            }
        )
        write_json(turn_dir / "batch_status.json", job_summary)
        append_json_list(scenario_root / "manifests" / "download_manifest.json", job_summary)
        summary["jobs"].append(job_summary)

    write_json(scenario_root / "summary" / "download_summary.json", summary)
    logger.info("download finished | scenario=%s", args.scenario_name)


if __name__ == "__main__":
    main()
