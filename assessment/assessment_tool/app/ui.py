"""UI helpers for rendering the annotation panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from schema import FIELD_ORDER_BY_PATH, LABELS, UNSET_LABEL  # type: ignore


def scrollable_container():
    """Fixed-height container so each column scrolls independently (Streamlit >= 1.33)."""
    try:
        return st.container(height=820)
    except TypeError:
        return st.container()


def init_field_state(
    state_key: str, field_paths: list[str], existing: list[dict[str, Any]] | None
) -> None:
    if state_key in st.session_state:
        return
    labels = {path: UNSET_LABEL for path in field_paths}
    notes = {path: "" for path in field_paths}
    if existing:
        by_path = {row.get("field_path"): row for row in existing}
        for path in field_paths:
            saved = by_path.get(path)
            if not saved:
                continue
            if bool(saved.get("label_was_explicitly_set", False)):
                labels[path] = saved.get("label", UNSET_LABEL)
            notes[path] = saved.get("note") or ""
    st.session_state[state_key] = {"labels": labels, "notes": notes}


def render_source_panel(md_text: str, md_path: str, pdf_path: str, paper_id: str) -> None:
    """Let the reviewer choose PDF (origin), rendered MD, or raw .md in the left column."""
    # st.markdown("### Source")
    mode = st.radio(
        "Text Source",
        ["PDF (origin)", "MD — rendered", "MD — raw"],
        horizontal=True,
        key=f"source_display::{paper_id}",
        label_visibility="collapsed",
    )
    # st.markdown(
    #     f"**MD:** `{md_path}`  \n**PDF (origin):** `{pdf_path}`  \n"
    #     f"_Pairing: `{Path(pdf_path).name}` ↔ `{Path(md_path).name}`_"
    # )

    pdf_file = Path(pdf_path)

    if mode == "PDF (origin)":
        if not pdf_file.is_file():
            st.warning(
                f"PDF not found at `{pdf_path}`. Expected `{pdf_file.name}` next to "
                f"`{Path(md_path).name}` under `vlm/`."
            )
            return
        pdf_bytes = pdf_file.read_bytes()
        pdf_viewer(pdf_bytes, width="100%", height=680)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=pdf_file.name,
            mime="application/pdf",
            key=f"pdf_dl::{paper_id}",
        )
        return

    if mode == "MD — rendered":
        st.markdown(md_text)
    else:
        st.text_area("Raw .md", md_text, height=620, key=f"md_raw::{paper_id}")


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if value is None:
        return "null"
    return str(value)


def _widget_key(kind: str, state_key: str, field_path: str) -> str:
    safe = (
        field_path.replace(".", "__")
        .replace("[", "_idx")
        .replace("]", "_")
        .replace(" ", "_")
    )
    return f"{kind}::{state_key}::{safe}"


def _normalize_order_path(path: str) -> str:
    if not path:
        return ""
    parts = path.split(".")
    normalized: list[str] = []
    for part in parts:
        if "[" in part:
            normalized.append(part.split("[", 1)[0] + "[]")
        else:
            normalized.append(part)
    return ".".join(normalized)


def _ordered_keys_for_path(path: str, keys: list[str]) -> list[str]:
    order = FIELD_ORDER_BY_PATH.get(_normalize_order_path(path), [])
    ordered = [name for name in order if name in keys]
    seen = set(ordered)
    remaining = [k for k in keys if k not in seen]
    return ordered + remaining


def _subtree_matches_filter(path: str, value: Any, q: str) -> bool:
    if not q:
        return True
    if q in path.lower():
        return True
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{path}.{k}" if path else k
            if _subtree_matches_filter(child, v, q):
                return True
        return False
    if isinstance(value, list):
        for i, item in enumerate(value):
            child = f"{path}[{i}]"
            if _subtree_matches_filter(child, item, q):
                return True
        return False
    return q in str(value).lower()


def _should_expand(path: str, value: Any, q: str) -> bool:
    if not q:
        return False
    if q in path.lower():
        return True
    if isinstance(value, dict):
        return any(
            _should_expand(f"{path}.{k}" if path else k, v, q) for k, v in value.items()
        )
    if isinstance(value, list):
        return any(_should_expand(f"{path}[{i}]", item, q) for i, item in enumerate(value))
    return False


def _render_leaf(
    state_key: str,
    field_path: str,
    value: Any,
    labels: dict[str, str],
    notes: dict[str, str],
    *,
    show_title: bool = True,
) -> None:
    if show_title:
        st.markdown(f"**`{field_path}`**")
    st.code(_display_value(value), wrap_lines=True)
    options = [UNSET_LABEL] + LABELS
    current = labels.get(field_path, UNSET_LABEL)
    try:
        idx = options.index(current)
    except ValueError:
        idx = 0
    selected = st.selectbox(
        f"Label: {field_path}",
        options=options,
        index=idx,
        format_func=lambda x: "" if x == UNSET_LABEL else x,
        key=_widget_key("label", state_key, field_path),
        label_visibility="collapsed",
    )
    labels[field_path] = selected
    notes[field_path] = st.text_area(
        "Field note (optional)",
        value=notes.get(field_path, ""),
        key=_widget_key("note", state_key, field_path),
        height=68,
    )


def _render_tree_node(
    state_key: str,
    path: str,
    value: Any,
    q: str,
    labels: dict[str, str],
    notes: dict[str, str],
) -> None:
    if not _subtree_matches_filter(path, value, q):
        return

    if isinstance(value, dict) and not path:
        if not value:
            return
        keys = _ordered_keys_for_path(path, list(value.keys()))
        matching_keys = [
            k for k in keys if _subtree_matches_filter(k, value[k], q)
        ]
        for k in matching_keys:
            _render_tree_node(state_key, k, value[k], q, labels, notes)
        return

    if isinstance(value, dict):
        if not value:
            return
        keys = _ordered_keys_for_path(path, list(value.keys()))
        matching_keys = [
            k
            for k in keys
            if _subtree_matches_filter(f"{path}.{k}" if path else k, value[k], q)
        ]
        if not matching_keys:
            return
        label = path or "root"
        expanded = _should_expand(path, value, q)
        with st.expander(label, expanded=expanded):
            for k in matching_keys:
                child_path = f"{path}.{k}" if path else k
                _render_tree_node(state_key, child_path, value[k], q, labels, notes)
        return

    if isinstance(value, list):
        label = path or "root"
        expanded = _should_expand(path, value, q)
        with st.expander(f"{label} ({len(value)} items)", expanded=expanded):
            _render_leaf(state_key, path, value, labels, notes, show_title=False)
        return

    _render_leaf(state_key, path, value, labels, notes)


def render_annotation_panel(state_key: str, extraction: dict[str, Any], filter_text: str) -> None:
    st.markdown("### Extraction + Annotation")
    data = st.session_state[state_key]
    labels: dict[str, str] = data["labels"]
    notes: dict[str, str] = data["notes"]

    q = filter_text.strip().lower()
    _render_tree_node(state_key, "", extraction, q, labels, notes)
    if q and not _subtree_matches_filter("", extraction, q):
        st.info("No fields match the current filter.")

