"""
Cross-paper entity canonicalization: given an entity mention from one paper's
extraction (name/type/organism/synonyms), resolve it to a stable canonical
ID shared with every other paper's mention of "the same" entity.

Without this, `Entity.local_id` in each paper's extraction.json is only
unique *within that paper* (see schema/sweet_transporter.py's docstring) —
paper A's "gene_1" and paper B's "gene_1" have no relationship. This module
is what makes graph nodes actually mergeable across the whole corpus.

Matching strategy, in order:
  1. Exact (case-insensitive) match against an existing canonical entity's
     name or any of its recorded aliases, scoped to the same `type` and (when
     both sides have one) the same `organism`.
  2. Fuzzy fallback via PaperReviewTool's `FuzzyNormalizer` (same difflib-based
     matcher used elsewhere in this codebase to snap strings onto a known
     label set) over that same candidate pool.
  3. No match -> a new canonical entity is created.

Organism scoping is deliberate: cross-species homologs (e.g. ZmSWEET4c vs.
OsSWEET4) must stay separate canonical nodes linked by an explicit
`is_homolog_of` edge (see DOMAIN_NOTES in schema/sweet_transporter.py) rather
than being silently merged just because their names are similar.

The canon table is a flat JSON list, persisted to `config.ENTITY_CANON_PATH`
and reloaded/saved on every call — fine at this corpus size; revisit if it
ever becomes a bottleneck.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as gene_config
from PaperReviewTool.src.normalizers import FuzzyNormalizer
from PaperSearch.utils import sanitize_filename

_fuzzy = FuzzyNormalizer(cutoff=gene_config.ENTITY_FUZZY_CUTOFF)


def load_canon() -> list[dict[str, Any]]:
    path = gene_config.ENTITY_CANON_PATH
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_canon(canon: list[dict[str, Any]]) -> None:
    path = gene_config.ENTITY_CANON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(canon, f, ensure_ascii=False, indent=2, default=str)


def _candidates(canon: list[dict[str, Any]], type_: str, organism: Optional[str]) -> list[dict[str, Any]]:
    out = []
    for entry in canon:
        if entry["type"] != type_:
            continue
        if organism and entry.get("organism") and entry["organism"].lower() != organism.lower():
            continue
        out.append(entry)
    return out


def _new_canonical_id(canon: list[dict[str, Any]], type_: str, name: str) -> str:
    base = f"{type_}_{sanitize_filename(name)}"
    existing_ids = {e["canonical_id"] for e in canon}
    if base not in existing_ids:
        return base
    i = 2
    while f"{base}_{i}" in existing_ids:
        i += 1
    return f"{base}_{i}"


def resolve_entity(
    name: str,
    type_: str,
    organism: Optional[str] = None,
    synonyms: Optional[list[str]] = None,
    source_doi: Optional[str] = None,
) -> str:
    """Resolve one entity mention to a canonical ID, updating (and persisting)
    the canon table with any new alias/provenance info in the process."""
    canon = load_canon()
    candidates = _candidates(canon, type_, organism)

    name_lower = name.strip().lower()
    match: Optional[dict[str, Any]] = None
    for entry in candidates:
        known = {entry["name"].lower(), *(a.lower() for a in entry.get("aliases", []))}
        if name_lower in known:
            match = entry
            break

    if match is None and candidates:
        pool: dict[str, dict[str, Any]] = {}
        for entry in candidates:
            pool[entry["name"]] = entry
            for alias in entry.get("aliases", []):
                pool.setdefault(alias, entry)
        snapped = _fuzzy.normalize(name, list(pool.keys()))
        if snapped is not None:
            match = pool[snapped]

    if match is not None:
        aliases = set(match.get("aliases", []))
        new_names = {name, *(synonyms or [])}
        if not new_names.issubset(aliases | {match["name"]}):
            match["aliases"] = sorted(aliases | new_names)
        if source_doi and source_doi not in match.get("source_dois", []):
            match.setdefault("source_dois", []).append(source_doi)
        save_canon(canon)
        return match["canonical_id"]

    canonical_id = _new_canonical_id(canon, type_, name)
    entry = {
        "canonical_id": canonical_id,
        "type": type_,
        "name": name,
        "organism": organism,
        "aliases": sorted(set(synonyms or [])),
        "source_dois": [source_doi] if source_doi else [],
    }
    canon.append(entry)
    save_canon(canon)
    return canonical_id
