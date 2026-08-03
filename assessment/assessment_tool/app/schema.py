"""Schema constants and typed structures for assessment annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LABELS = [
    "correct",
    "missing_critical_component",
    "missing_minor_component",
    "partial_wrong",
    "missing_critical_component_and_partial_wrong",
    "missing_minor_component_and_partial_wrong",
    "completely_wrong",
]

DEFAULT_LABEL = "correct"
UNSET_LABEL = "__UNSET__"

NAME_FIELDS = [
    ["metadata", "relevance"],
    ["context"],
    ["scale", "methodology", "tile_design"],
    ["results", "internal_evaluation"],
    ["key_words", "overall_synthesis"],
]

ROOT_FIELD_ORDER = [
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
    "patch_set",
]

FIELD_ORDER_BY_PATH = {
    "": ROOT_FIELD_ORDER,
    "metadata": ["title", "study_type", "location"],
    "metadata.location": [
        "country",
        "state_province",
        "county",
        "watershed",
        "coordinates_text",
        "spatial_extent",
    ],
    "relevance": ["level", "is_primary", "justification"],
    "context": ["scientific_question", "research_gap", "objectives", "hypotheses"],
    "scale": [
        "category",
        "total_area_value",
        "total_area_unit",
        "area_description",
        "site_count",
        "temporal_resolution",
    ],
    "methodology": [
        "study_duration",
        "approach_types",
        "physical_models",
        "ml_models",
        "analytical_methods",
        "key_assumptions",
        "extraction_FLAG",
    ],
    "tile_design": [
        "depth",
        "spacing",
        "pipe_diameter",
        "pipe_material",
        "drainage_coefficient",
        "soil_texture",
        "drainage_practices",
        "associated_management_practices",
        "other_practices",
        "extraction_FLAG",
    ],
    "results": ["summary_bullets", "quantitative_stats", "extraction_FLAG"],
    "results.quantitative_stats[]": ["variable", "value", "unit", "context"],
    "internal_evaluation": [
        "highlights",
        "limitations",
        "future_directions",
        "extraction_FLAG",
    ],
    "overall_synthesis": ["main_contribution", "executive_summary", "extraction_FLAG"],
}


@dataclass(frozen=True)
class FieldAnnotation:
    field_path: str
    label: str
    label_was_explicitly_set: bool
    note: Optional[str] = None

