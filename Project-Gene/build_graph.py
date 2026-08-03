#!/usr/bin/env python3
"""
Step 4: aggregate every paper's extraction.json into one corpus-level graph.

For each manifest row with extraction_status="done": load its extraction.json,
resolve each entity to a canonical ID via `entity_resolution.resolve_entity()`
(building/updating the shared canon table as it goes), remap each relation's
subject_id/object_id from paper-local IDs to canonical IDs, and write:

  - graph/nodes.json: one row per canonical entity (the canon table itself)
  - graph/edges.json: one row per relation, with canonical `source`/`target`
    ids plus all of Relation's original fields (mechanism, evidence,
    growth_stage, ...) and which paper it came from. Multiple papers
    reporting the same gene-phenotype pair produce multiple edges — they are
    NOT merged into one, so each claim keeps its own evidence/provenance
    instead of being blended away.

Plain JSON, not a networkx object — trivially loadable into one:
    import json, networkx as nx
    nodes = json.load(open("graph/nodes.json"))
    edges = json.load(open("graph/edges.json"))
    g = nx.MultiDiGraph()
    g.add_nodes_from((n["canonical_id"], n) for n in nodes)
    g.add_edges_from((e["source"], e["target"], e) for e in edges)

Rerun any time step 3 produces new extractions — rebuilds nodes/edges from
scratch off the current canon table + every extraction.json each time (cheap
at this corpus size). The canon table (entity_resolution's persisted state)
is what actually carries information forward between runs.

Usage (from repo root, `llmReview` conda env active):
    python Project-Gene/build_graph.py
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as gene_config
import entity_resolution
from PaperSearch.manifest import load_manifest

logger = logging.getLogger("project_gene.build_graph")

# Node fill color by entity type — kept distinct so gene/phenotype/etc. are
# visually separable at a glance without reading every label.
_NODE_COLORS = {
    "Gene": "#6BAED6",
    "Protein": "#9E9AC8",
    "Metabolite_Substrate": "#FDB863",
    "Phenotype_Trait": "#FB6A4A",
    "Tissue_CellType": "#74C476",
}
_DEFAULT_NODE_COLOR = "#CCCCCC"


def _load_extraction(row: dict[str, Any]) -> dict[str, Any] | None:
    path = row.get("extraction_path")
    if not path or not Path(path).is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_manifest(gene_config.MANIFEST_PATH)
    done_rows = [r for r in rows if r.get("extraction_status") == "done"]

    edges: list[dict[str, Any]] = []
    papers_processed = 0
    for row in done_rows:
        doi = row["doi"]
        extraction = _load_extraction(row)
        if extraction is None:
            logger.warning("skipping doi=%s: extraction_path missing/unreadable", doi)
            continue

        entities = extraction.get("entities") or []
        relations = extraction.get("relations") or []
        if not entities:
            continue

        local_to_canonical: dict[str, str] = {}
        for ent in entities:
            canonical_id = entity_resolution.resolve_entity(
                name=ent["name"],
                type_=ent["type"],
                organism=ent.get("organism"),
                synonyms=ent.get("synonyms"),
                source_doi=doi,
            )
            local_to_canonical[ent["local_id"]] = canonical_id

        for rel in relations:
            subject = local_to_canonical.get(rel["subject_id"])
            target = local_to_canonical.get(rel["object_id"])
            if subject is None or target is None:
                logger.warning(
                    "doi=%s: relation references unknown local_id (subject=%s, object=%s); skipping",
                    doi, rel.get("subject_id"), rel.get("object_id"),
                )
                continue
            edge = dict(rel)
            edge["source"] = subject
            edge["target"] = target
            edge["source_doi"] = doi
            edges.append(edge)

        papers_processed += 1

    nodes = entity_resolution.load_canon()
    logger.info("built graph | papers=%d | nodes=%d | edges=%d", papers_processed, len(nodes), len(edges))
    return nodes, edges


def save(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    out_dir = gene_config.GRAPH_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nodes.json").open("w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2, default=str)
    with (out_dir / "edges.json").open("w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2, default=str)


def _node_tooltip(node: dict[str, Any]) -> str:
    lines = [
        f"<b>{html.escape(node['name'])}</b> ({html.escape(node['type'])})",
    ]
    if node.get("organism"):
        lines.append(f"organism: {html.escape(node['organism'])}")
    aliases = [a for a in node.get("aliases", []) if a != node["name"]]
    if aliases:
        lines.append(f"aliases: {html.escape(', '.join(aliases))}")
    n_papers = len(node.get("source_dois", []))
    lines.append(f"seen in {n_papers} paper(s)")
    return "<br>".join(lines)


def _edge_tooltip(edge: dict[str, Any]) -> str:
    lines = [f"<b>{html.escape(edge.get('relation_type', ''))}</b>"]
    for field in ("mechanism", "evidence_type", "effect_direction", "quantitative_effect"):
        if edge.get(field):
            lines.append(f"{field}: {html.escape(str(edge[field]))}")
    if edge.get("growth_stage"):
        lines.append(f"growth_stage: {html.escape(', '.join(edge['growth_stage']))}")
    if edge.get("evidence"):
        lines.append(f"evidence: &ldquo;{html.escape(edge['evidence'])}&rdquo;")
    lines.append(f"source: {html.escape(edge.get('source_doi', ''))}")
    return "<br>".join(lines)


def _legend_html(types_present: set[str]) -> str:
    items = []
    for t in sorted(types_present):
        color = _NODE_COLORS.get(t, _DEFAULT_NODE_COLOR)
        items.append(
            '<span style="margin-right:24px;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{color};margin-right:6px;vertical-align:middle;"></span>'
            f'{html.escape(t)}</span>'
        )
    return (
        '<div style="padding:10px 16px;background:#fafafa;border-bottom:1px solid #ddd;'
        'font-family:sans-serif;font-size:14px;">'
        '<b style="margin-right:16px;">Legend:</b>' + "".join(items) +
        "</div>"
    )


def draw(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], out_path: Path) -> None:
    """Render an interactive HTML graph (pyvis/vis.js): node/edge labels stay
    short (entity name / relation_type) and the full mechanism, evidence,
    growth stage, merged aliases, etc. show up on hover — cramming that text
    directly onto the diagram would just overlap into an unreadable mess once
    there's more than a handful of edges. A color legend for node types sits
    at the top of the page."""
    from pyvis.network import Network

    net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut()

    node_ids = {n["canonical_id"] for n in nodes}
    for n in nodes:
        net.add_node(
            n["canonical_id"],
            label=n["name"],
            title=_node_tooltip(n),
            color=_NODE_COLORS.get(n["type"], _DEFAULT_NODE_COLOR),
            shape="dot",
            size=12 + 3 * len(n.get("source_dois", [])),
        )

    skipped = 0
    for e in edges:
        if e["source"] not in node_ids or e["target"] not in node_ids:
            skipped += 1
            continue
        net.add_edge(
            e["source"],
            e["target"],
            label=e.get("relation_type", ""),
            title=_edge_tooltip(e),
            arrows="to",
        )
    if skipped:
        logger.warning("draw: skipped %d edge(s) referencing unknown node ids", skipped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False, open_browser=False)

    # pyvis has no built-in legend; inject a plain HTML/CSS bar above the
    # network canvas rather than faking it with un-draggable dummy nodes.
    legend = _legend_html({n["type"] for n in nodes})
    content = out_path.read_text(encoding="utf-8")
    out_path.write_text(content.replace("<body>", "<body>\n" + legend, 1), encoding="utf-8")


def load_saved_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load nodes.json/edges.json previously written by `save()`, without
    rebuilding them (skips re-walking the manifest / re-resolving entities)."""
    out_dir = gene_config.GRAPH_OUTPUT_DIR
    nodes_path, edges_path = out_dir / "nodes.json", out_dir / "edges.json"
    if not nodes_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError(
            f"{nodes_path} / {edges_path} not found — run `python Project-Gene/build_graph.py` "
            "(without --draw-only) at least once first."
        )
    with nodes_path.open("r", encoding="utf-8") as f:
        nodes = json.load(f)
    with edges_path.open("r", encoding="utf-8") as f:
        edges = json.load(f)
    return nodes, edges


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-paper extractions into one corpus-level graph.")
    p.add_argument(
        "--draw", action="store_true",
        help="Also render an interactive HTML visualization (graph/graph.html) — hover a node/edge for details.",
    )
    p.add_argument(
        "--draw-only", action="store_true",
        help="Skip rebuilding nodes.json/edges.json; just re-render graph.html from what's already on disk.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    if args.draw_only:
        nodes, edges = load_saved_graph()
    else:
        nodes, edges = build()
        save(nodes, edges)
        print(f"nodes={len(nodes)} edges={len(edges)} -> {gene_config.GRAPH_OUTPUT_DIR}")

    if args.draw or args.draw_only:
        html_path = gene_config.GRAPH_OUTPUT_DIR / "graph.html"
        draw(nodes, edges, html_path)
        print(f"drew graph -> {html_path}")
