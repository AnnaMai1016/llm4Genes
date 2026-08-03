"""Streamlit MVP for manual extraction assessment."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from data_loader import (  # type: ignore
    CASE_MAP,
    extraction_path,
    flatten_fields,
    list_cases,
    list_paper_ids,
    load_extraction,
    load_md,
    paper_md_path,
    paper_pdf_origin_path,
)
from schema import UNSET_LABEL  # type: ignore
from storage import (  # type: ignore
    find_record,
    load_all_records,
    normalize_annotations,
    now_iso,
    save_record,
)
from ui import (  # type: ignore
    init_field_state,
    render_annotation_panel,
    render_source_panel,
    scrollable_container,
)


TOOL_VERSION = "v0.1.0"


def _state_key(paper_id: str, case_id: str, reviewer_id: str) -> str:
    return f"ann::{reviewer_id}::{paper_id}::{case_id}"


def _completion_ratio(records: list[dict], reviewer_id: str, papers: list[str], cases: list[str]) -> tuple[int, int]:
    total = len(papers) * len(cases)
    completed = 0
    for paper_id in papers:
        for case_id in cases:
            rec = find_record(records, paper_id, case_id, reviewer_id)
            if rec and rec.get("is_complete", False):
                completed += 1
    return completed, total


def main() -> None:
    st.set_page_config(
        # page_title="Assessment Tool", 
        layout="wide"
        )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 0.6rem;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextInput"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stSelectbox"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stRadio"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stCheckbox"]) {
            margin-bottom: 0.3rem;
        }
        div[data-testid="stMetric"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # st.title("Assessment Tool")

    cases = list_cases()
    papers = list_paper_ids()
    if not papers:
        st.error("No paper .md files found under `assessment/samples/mineru/*/vlm/`.")
        return
    if not cases:
        st.error("No cases found under `assessment/runs/*/results`.")
        return

    records = load_all_records()

    with st.sidebar:
        st.markdown("### Controls")
        reviewer_id = st.text_input("Reviewer", value="ma7")
        paper_id = st.selectbox("Paper ID", options=papers)
        case_id = st.selectbox(
            "Case",
            options=cases,
            format_func=lambda c: f"{c} - {CASE_MAP.get(c, c)}",
        )
        completed, total = _completion_ratio(records, reviewer_id, papers, cases)
        st.metric("Progress", f"{completed}/{total}")

    state_key = _state_key(paper_id, case_id, reviewer_id)
    current = find_record(records, paper_id, case_id, reviewer_id)
    existing_fields = (current or {}).get("field_annotations", [])

    try:
        md_text = load_md(paper_id)
    except Exception as exc:
        st.error(str(exc))
        return
    try:
        extraction = load_extraction(case_id, paper_id)
    except Exception as exc:
        st.error(str(exc))
        return

    flat_fields = flatten_fields(extraction)
    flat_paths = [path for path, _ in flat_fields]
    init_field_state(state_key, flat_paths, existing_fields)

    with st.sidebar:
        st.markdown("---")
        filter_text = st.text_input("Filter field path", value="", placeholder="e.g., methodology")
        saved_at = (current or {}).get("reviewed_at")
        status = f"Loaded previous review at {saved_at}" if saved_at else "No saved review yet"
        st.caption(status)
        is_complete = st.checkbox("Mark as complete", value=bool((current or {}).get("is_complete", False)))
        save_clicked = st.button("Save review", type="primary", use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        with scrollable_container():
            md_path = str(Path(paper_md_path(paper_id)).as_posix())
            pdf_path = str(Path(paper_pdf_origin_path(paper_id)).as_posix())
            render_source_panel(md_text, md_path, pdf_path, paper_id)
    with right:
        with scrollable_container():
            render_annotation_panel(state_key, extraction, filter_text)
            global_note = st.text_area(
                "Global note (optional)",
                value=(current or {}).get("global_note", ""),
                key=f"global_note::{state_key}",
            )

    data = st.session_state[state_key]
    if save_clicked:
        bundled = normalize_annotations(flat_paths, data["labels"], data["notes"])
        record = {
            "paper_id": paper_id,
            "case_id": case_id,
            "reviewer_id": reviewer_id,
            "reviewed_at": now_iso(),
            "global_note": (global_note or "").strip() or None,
            "field_annotations": bundled,
            "is_complete": bool(is_complete),
            "source_md_path": str(Path(paper_md_path(paper_id)).as_posix()),
            "source_pdf_origin_path": str(Path(paper_pdf_origin_path(paper_id)).as_posix()),
            "extraction_path": str(Path(extraction_path(case_id, paper_id)).as_posix()),
            "tool_version": TOOL_VERSION,
        }
        save_record(record)
        st.success(
            "Saved. Unselected fields were stored as `correct` "
            "with `label_was_explicitly_set=false`."
        )

    with st.expander("Raw extraction JSON", expanded=False):
        st.code(json.dumps(extraction, ensure_ascii=False, indent=2))
    with st.expander("Current annotation state (debug)", expanded=False):
        preview = {
            "labels_count": len(data.get("labels", {})),
            "unset_count": sum(1 for v in data.get("labels", {}).values() if v == UNSET_LABEL),
        }
        st.json(preview)


if __name__ == "__main__":
    main()

