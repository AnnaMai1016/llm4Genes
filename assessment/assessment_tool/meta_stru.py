"""
Pydantic schema for grounded paper extraction in tile drainage studies.

Nullability convention
----------------------
Every field is ``Optional`` so a patch can explicitly set it to ``null``:
  - ``None`` (null) = explicitly cleared / unknown / should be absent.
  - ``[]``          = extracted and confirmed empty list.
  - ``[...]``       = has values.

Sub-section fields in ``ExtractionSchema`` default to ``None`` (not
``default_factory``) for the same reason: a whole section can be nulled out.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from ..patches import PatchSet

class ExtractionBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

NAME_FIELDS = [
    ['metadata', 'relevance'],
    ['context'],
    ['scale','methodology','tile_design'],
    ['results','internal_evaluation',],
    ['key_words','overall_synthesis',]
]

VERIFICATION_PREFIXES = [[
    "/relevance",
    "/location",
    "/context",
    "/scale",
    "/methodology",
    "/tile_design",
    "/results/summary_bullets",
    "/internal_evaluation",
]]

# -----------------------------
# Relevance
# -----------------------------

class Relevance(ExtractionBaseModel):
    level: Optional[
        Literal[
            "Tile-centered",
            "Tile-related",
            "Mentioned",
            "Little relevance",
            "Not relevant",
        ]
    ] = Field(
        default=None,
        description="Overall relevance of the paper to tile drainage.",
    )
    is_primary: Optional[bool] = Field(
        default=None,
        description="True if tile drainage is the primary focus of the study.",
    )
    justification: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim quote from the paper supporting the relevance judgment. "
            "The exact text or paraphrased text."
        ),
    )


# -----------------------------
# Metadata / Location
# -----------------------------

class Location(ExtractionBaseModel):
    country: Optional[str] = Field(default=None, description="Country of the study focus, can be Global or specific countries")
    state_province: Optional[str] = None
    county: Optional[str] = None
    watershed: Optional[str] = None
    coordinates_text: Optional[str] = Field(
        default=None,
        description="Coordinates exactly as stated in the paper, if available.",
    )
    spatial_extent: Optional[str] = Field(
        default=None,
        description="Textual description of the study area extent.",
    )


class Metadata(ExtractionBaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Title of the article.",
    )
    study_type: Optional[
        List[
            Literal[
                "Observational",        # field / survey / historical data analysis
                "Experimental",         # controlled experiments (field or lab)
                "Process_Modeling",     # mechanistic models (e.g., ecosys, hydrology models)
                "Data_Driven_Modeling", # ML/statistical models (RF, regression, DL)
                "Remote_Sensing",       # satellite-based mapping / inference
                "Method_Development",   # new methods, algorithms, datasets
                "Review"                # synthesis / review papers
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "Study type(s) describing the methodological approach. "
            "Use multiple labels for hybrid studies (e.g., ['Observational', "
            "'Process_Modeling']). "
            "Definitions: Observational = analysis of existing field or survey data; "
            "Experimental = controlled manipulation; Process_Modeling = mechanistic "
            "simulation models; Data_Driven_Modeling = statistical or machine learning models; "
            "Remote_Sensing = satellite or aerial data-based analysis; "
            "Method_Development = new methods or datasets; Review = literature synthesis."
        ),
    )
    location: Optional[Location] = Field(
        default=None,
        description="Geographic location of the study. Null to clear the entire location block.",
    )


# -----------------------------
# Context
# -----------------------------

class Context(ExtractionBaseModel):
    scientific_question: Optional[str] = Field(
        default=None,
        description=(
            "Explicitly stated scientific question(s) only, not inferred objectives. "
            "Captures the 'why/what' framing — the knowledge gap being interrogated "
            "(e.g., 'Does controlled drainage reduce nitrate loads?'). "
            "Distinct from objectives, which describe what the study will do."
        ),
    )
    research_gap: Optional[str] = Field(
        default=None,
        description="Explicitly stated research gap or unmet need.",
    )
    objectives: Optional[str] = Field(
        default=None,
        description=(
            "Explicitly stated study objectives or aims. "
            "Captures the 'we will measure/test/model' framing — concrete actions "
            "the study undertakes (e.g., 'to quantify annual nitrate export under three drainage treatments'). "
            "Distinct from scientific_question, which frames the broader knowledge gap."
        ),
    )
    hypotheses: Optional[List[str]] = Field(
        default=None,
        description="Explicitly stated hypotheses only. Null to clear; [] if confirmed none.",
    )


# -----------------------------
# Scale
# -----------------------------

class Scale(ExtractionBaseModel):
    category: Optional[
        Literal[
            "Site",
            # "Multiple Sites",
            "Watershed",
            "Regional",
            "Unclear"
        ]
    ] = Field(
        default=None,
        description="Spatial scale of the study. Site: Individual plots or fields (even if multiple locations). Watershed: Study targets at a specific watershed. Regional: Targeting at a specific region. Unclear: Not clear from the paper.",
    )
    total_area_value: Optional[float] = Field(
        default=None,
        description="Numeric total study area if explicitly stated.",
    )
    total_area_unit: Optional[Literal["ha", "m2", "km2"]] = Field(
        default=None,
        description="Unit for total_area_value.",
    )
    area_description: Optional[str] = Field(
        default=None,
        description="Qualitative or approximate description of area if numeric area is unavailable.",
    )
    site_count: Optional[int] = Field(
        default=None,
        description="Number of sites explicitly stated, if applicable.",
    )
    temporal_resolution: Optional[str] = Field(
        default=None,
        description=(
            "Measurement or reporting frequency as stated in the paper "
            "(e.g., 'sub-daily', 'event-based', 'monthly', 'annual'). "
            "Distinct from study_duration in Methodology, which captures total time span."
        ),
    )


# -----------------------------
# Methodology
# -----------------------------

class Methodology(ExtractionBaseModel):
    study_duration: Optional[str] = Field(
        default=None,
        description="Study duration exactly or closely as stated, including units or time span.",
    )
    approach_types: Optional[List[
        Literal[
            "Physical Model Development",
            "Physical Model Application",
            "Conceptual Model",
            "Machine Learning Model",
            "Experimental",
            "Observational",
            "Remote Sensing"
        ]
    ]] = Field(
        default=None,
        description="High-level methodological approach categories explicitly supported by the paper. Null to clear; [] if confirmed none.",
    )
    physical_models: Optional[List[str]] = Field(
        default=None,
        description="Specific process-based or physics-based models used, if any. Null to clear; [] if confirmed none.",
    )
    ml_models: Optional[List[str]] = Field(
        default=None,
        description="Specific machine learning models or algorithms used, if any. Null to clear; [] if confirmed none.",
    )
    analytical_methods: Optional[List[str]] = Field(
        default=None,
        description="Explicit analytical, statistical, or computational methods used. Null to clear; [] if confirmed none.",
    )
    key_assumptions: Optional[List[str]] = Field(
        default=None,
        description="Explicitly stated methodological assumptions. Null to clear; [] if confirmed none.",
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for Methodology",
    )


# -----------------------------
# Tile design / management
# -----------------------------

class TileDesign(ExtractionBaseModel):
    depth: Optional[str] = Field(
        default=None,
        description="Tile drainage depth exactly as stated.",
    )
    spacing: Optional[str] = Field(
        default=None,
        description="Tile spacing exactly as stated.",
    )
    pipe_diameter: Optional[str] = Field(
        default=None,
        description="Pipe diameter exactly as stated (e.g., '100 mm').",
    )
    pipe_material: Optional[str] = Field(
        default=None,
        description="Pipe material explicitly stated (e.g., 'corrugated plastic', 'clay tile').",
    )
    drainage_coefficient: Optional[str] = Field(
        default=None,
        description="Drainage coefficient exactly as stated.",
    )
    soil_texture: Optional[str] = Field(
        default=None,
        description="Soil texture explicitly mentioned in relation to tile drainage.",
    )
    drainage_practices: Optional[List[
        Literal[
            "No Drainage",
            "Controlled Drainage",
            "Bioreactor",
            "Conventional Drainage",
            "Saturated Buffer",
            "Subirrigation",
        ]
    ]] = Field(
        default=None,
        description=(
            "Tile-drainage-related practices **actually implemented or observed at the study site(s)** "
            "during the study period. Do NOT include practices merely cited, reviewed, discussed as "
            "alternatives, or mentioned in introduction/background/discussion sections without being "
            "applied in this study. "
            "No Drainage means without tile drainage. Free Drainage is treated as Conventional Drainage. "
            "Null to clear; [] if confirmed none."
        ),
    )
    associated_management_practices: Optional[List[
        Literal[
            "Cover Crop",
            "Fertilizer Treatment",
            "Tillage",
            "Wetland/Filter Strip",
            "Irrigation",
            "Crop Rotation",
        ]
    ]] = Field(
        default=None,
        description=(
            "Non-drainage management practices **actually implemented or observed at the study site(s)** "
            "during the study period. Do NOT include practices merely cited, reviewed, discussed as "
            "alternatives, or mentioned in introduction/background/discussion sections without being "
            "applied in this study. "
            "Null to clear; [] if confirmed none."
        ),
    )
    other_practices: Optional[List[str]] = Field(
        default=None,
        description=(
            "Other practices actually implemented or observed at the study site(s) during the study period "
            "explicitly stated that are not covered by the predefined categories. "
            "Null to clear; [] if confirmed none."
        ),
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for TileDesign",
    )


# -----------------------------
# Results
# -----------------------------

class StatEntry(ExtractionBaseModel):
    variable: Optional[str] = Field(
        default=None,
        description="Reported variable name.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Exact reported value as text.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit associated with the value, if explicitly stated.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Brief contextual note, such as treatment, comparison, or condition tied to the statistic.",
    )


class Results(ExtractionBaseModel):
    summary_bullets: Optional[List[str]] = Field(
        default=None,
        description="Main findings explicitly grounded in the paper. Null to clear; [] if confirmed none.",
    )
    quantitative_stats: Optional[List[StatEntry]] = Field(
        default=None,
        description="Important quantitative results central to the study's conclusions. Null to clear; [] if confirmed none.",
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for Results",
    )


# -----------------------------
# Author-reported evaluation
# -----------------------------

class InternalEvaluation(ExtractionBaseModel):
    highlights: Optional[List[str]] = Field(
        default=None,
        description="Author-claimed strengths, achievements, or notable contributions explicitly stated in the paper. Null to clear; [] if confirmed none.",
    )
    limitations: Optional[List[str]] = Field(
        default=None,
        description="Explicitly stated study limitations. Null to clear; [] if confirmed none.",
    )
    future_directions: Optional[List[str]] = Field(
        default=None,
        description="Explicitly stated future work or recommended next steps. Null to clear; [] if confirmed none.",
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for InternalEvaluation",
    )


# -----------------------------
# Higher-level synthesis
# -----------------------------

class OverallSynthesis(ExtractionBaseModel):
    main_contribution: Optional[str] = Field(
        default=None,
        description="High-level summary of the work's contribution or novelty.",
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description="Short 2-3 sentence overview of objectives, methods, and conclusions.",
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for OverallSynthesis",
    )

# -----------------------------
# Optional glossary
# -----------------------------

class GlossaryItem(ExtractionBaseModel):
    term: Optional[str] = Field(
        default=None,
        description="Technical term appearing in the paper.",
    )
    definition: Optional[str] = Field(
        default=None,
        description="Definition based on the paper context.",
    )
    extraction_FLAG: Optional[bool] = Field(
        default=None,
        description="True if process extraction for GlossaryItem",
    )

# -----------------------------
# Root schema
# -----------------------------

class ExtractionSchema(ExtractionBaseModel):
    metadata: Optional[Metadata] = Field(
        default=None,
        description="Paper metadata and location. Null to clear the entire section.",
    )
    relevance: Optional[Relevance] = Field(
        default=None,
        description="Relevance assessment. Null to clear.",
    )
    key_words: Optional[List[str]] = Field(
        default=None,
        description=(
            "High-level scientific concepts and domain themes, ideally 10-15 items. "
            "Use broad themes rather than narrow site names or exact numeric values. "
            "Null to clear; [] if confirmed none."
        ),
    )
    context: Optional[Context] = Field(
        default=None,
        description="Research context. Null to clear the entire section.",
    )
    scale: Optional[Scale] = Field(
        default=None,
        description="Spatial and temporal scale. Null to clear the entire section.",
    )
    methodology: Optional[Methodology] = Field(
        default=None,
        description="Methodology details. Null to clear the entire section.",
    )
    tile_design: Optional[TileDesign] = Field(
        default=None,
        description="Tile design and management used in the studies. Null to clear the entire section.",
    )
    results: Optional[Results] = Field(
        default=None,
        description="Study results. Null to clear the entire section.",
    )
    internal_evaluation: Optional[InternalEvaluation] = Field(
        default=None,
        description="Author-reported evaluation. Null to clear the entire section.",
    )
    overall_synthesis: Optional[OverallSynthesis] = Field(
        default=None,
        description="Overall synthesis. Null to clear the entire section.",
    )
    patch_set: Optional[PatchSet] = Field(
        default=None,
        description="Only used for verification and updation process.")

    # technical_glossary: List[GlossaryItem] = Field(default_factory=list)
    # Retained for future use; enables per-paper term definitions for domain glossary building.
