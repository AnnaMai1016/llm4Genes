"""Storage helpers for bundled `(paper_id, case_id)` assessment records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schema import DEFAULT_LABEL, LABELS, UNSET_LABEL  # type: ignore


def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def annotation_file() -> Path:
    return data_dir() / "annotations.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_all_records() -> list[dict[str, Any]]:
    path = annotation_file()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_record(
    records: list[dict[str, Any]], paper_id: str, case_id: str, reviewer_id: str
) -> dict[str, Any] | None:
    for rec in reversed(records):
        if (
            rec.get("paper_id") == paper_id
            and rec.get("case_id") == case_id
            and rec.get("reviewer_id") == reviewer_id
        ):
            return rec
    return None


def save_record(record: dict[str, Any]) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    path = annotation_file()
    records = load_all_records()
    records = [
        existing
        for existing in records
        if not (
            existing.get("paper_id") == record.get("paper_id")
            and existing.get("case_id") == record.get("case_id")
            and existing.get("reviewer_id") == record.get("reviewer_id")
        )
    ]
    records.append(record)

    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_annotations(
    field_paths: list[str], selected_labels: dict[str, str], notes: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = set(LABELS)
    for path in field_paths:
        selected = selected_labels.get(path, UNSET_LABEL)
        raw_note = notes.get(path, "")
        note = raw_note.strip() or None
        if selected == UNSET_LABEL:
            label = DEFAULT_LABEL
            explicit = False
        else:
            label = selected if selected in valid else DEFAULT_LABEL
            explicit = True
        rows.append(
            {
                "field_path": path,
                "label": label,
                "label_was_explicitly_set": explicit,
                "note": note,
            }
        )
    return rows

