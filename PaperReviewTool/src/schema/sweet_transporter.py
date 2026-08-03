"""
Pydantic schema for grounded entity-relation extraction of genotype-phenotype
mechanisms, drafted for the maize/rice SWEET sugar-transporter gene family.

Unlike `schema/meta_stru.py` (one flat per-paper summary), this schema is
entity/relation-shaped so that per-paper extractions can be aggregated across
a corpus into a knowledge graph: `entities` are graph nodes (genes, proteins,
phenotypes, ...) and `relations` are edges between them, each carrying its own
grounded evidence and mechanism.

Nullability convention (same as schema/meta_stru.py)
------------------------------------------------------
Every field is ``Optional`` so a patch can explicitly set it to ``null``:
  - ``None`` (null) = explicitly cleared / unknown / should be absent.
  - ``[]``          = extracted and confirmed empty list.
  - ``[...]``       = has values.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from ..patches import PatchSet
from ..prompt import ExtractionPrompt


class ExtractionBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NAME_FIELDS = [
    ['metadata'],
    ['entities', 'relations'],
]
# entities and relations MUST be extracted in the same call (not split into
# separate turns): `Relation.subject_id`/`object_id` reference `Entity.local_id`
# from the same response, and each turn is a stateless API call that never
# sees a previous turn's output — split them and the model invents two
# unrelated local_id numbering schemes that don't line up.

VERIFICATION_PREFIXES = [[
    "/entities",
    "/relations",
]]


# -----------------------------
# Metadata
# -----------------------------

class Metadata(ExtractionBaseModel):
    title: Optional[str] = Field(default=None, description="Title of the article.")
    organism: Optional[List[str]] = Field(
        default=None,
        description=(
            "Species with primary experimental data in this paper (e.g. 'Zea mays', "
            "'Oryza sativa'). Do not include species only mentioned in passing/background. "
            "Null to clear; [] if confirmed none."
        ),
    )


# -----------------------------
# Entities (genes, proteins, phenotypes, tissues, substrates)
# -----------------------------

class Entity(ExtractionBaseModel):
    local_id: str = Field(
        ...,
        description=(
            "Paper-scoped identifier for referencing this entity from `relations` "
            "(e.g. 'gene_1', 'phenotype_1'). Not a global database ID — unique only "
            "within this paper's extraction."
        ),
    )
    name: str = Field(
        ...,
        description="Entity name exactly as it appears in the paper (e.g. 'ZmSWEET4c', 'grain weight').",
    )
    type: Literal[
        "Gene",
        "Protein",
        "Metabolite_Substrate",
        "Phenotype_Trait",
        "Tissue_CellType",
    ] = Field(..., description="Entity category.")
    organism: Optional[str] = Field(
        default=None,
        description="Species this entity belongs to (mainly for Gene/Protein entities).",
    )
    synonyms: Optional[List[str]] = Field(
        default=None,
        description=(
            "Other names/aliases used for this entity in the paper, including cross-species "
            "homolog gene symbols the paper explicitly equates to it (e.g. 'OsSWEET4' as a "
            "homolog of 'ZmSWEET4c'). Null to clear; [] if confirmed none."
        ),
    )
    expression_location: Optional[List[str]] = Field(
        default=None,
        description=(
            "Tissue, cell type, or subcellular compartment where the gene/protein is expressed "
            "or localized, exactly as stated (e.g. 'basal endosperm transfer layer (BETL)', "
            "'plasma membrane', 'scutellum'). This is NOT chromosomal/genomic location. "
            "Only meaningful for Gene/Protein entities. Null to clear; [] if confirmed none."
        ),
    )
    transported_substrate: Optional[List[
        Literal["Sucrose", "Glucose", "Fructose", "Hexose_Unspecified", "Other_Sugar"]
    ]] = Field(
        default=None,
        description=(
            "Sugar substrate(s) this gene/protein transports, if explicitly stated. "
            "Only meaningful for Gene/Protein entities acting as transporters. "
            "Null to clear; [] if confirmed none."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief grounded description of the entity as characterized in the paper.",
    )
    extraction_FLAG: Optional[bool] = Field(default=None, description="True if process extraction for Entity")


# -----------------------------
# Relations (genotype-phenotype / gene-gene / gene-substrate edges)
# -----------------------------

class Relation(ExtractionBaseModel):
    subject_id: str = Field(..., description="`local_id` of the source entity (e.g. the gene).")
    object_id: str = Field(..., description="`local_id` of the target entity (e.g. the phenotype).")
    relation_type: Literal[
        "causes",
        "positively_regulates",
        "negatively_regulates",
        "is_required_for",
        "transports",
        "is_homolog_of",
        "associated_with",
    ] = Field(..., description="Type of relationship between subject and object.")
    mechanism: Optional[str] = Field(
        default=None,
        description=(
            "Grounded description of the biological mechanism connecting subject and object, "
            "as stated in the paper (e.g. how transporter activity leads to the observed phenotype)."
        ),
    )
    growth_stage: Optional[List[
        Literal[
            "Germination",
            "Vegetative_Growth",
            "Seed_Development",
            "Reproductive_Stage",
            "Stress_Response",
            "Unclear",
        ]
    ]] = Field(
        default=None,
        description="Developmental/growth stage(s) during which this relation was observed to hold. Null to clear; [] if confirmed none.",
    )
    growth_stage_detail: Optional[str] = Field(
        default=None,
        description=(
            "Finer-grained stage description if the paper is more specific than the "
            "growth_stage categories (e.g. 'grain filling, 15 DAP')."
        ),
    )
    evidence_type: Optional[Literal[
        "Knockout_Knockdown",
        "Overexpression",
        "Transgenic",
        "Expression_Profiling",
        "GWAS",
        "QTL_Mapping",
        "Biochemical_Assay",
        "Computational_Prediction",
    ]] = Field(
        default=None,
        description="Experimental method that supports this relation.",
    )
    effect_direction: Optional[Literal["positive", "negative", "no_effect", "context_dependent"]] = Field(
        default=None,
        description=(
            "Direction of the subject's effect on the object, if applicable "
            "(e.g. knockout reduces grain weight -> subject positively contributes to object)."
        ),
    )
    quantitative_effect: Optional[str] = Field(
        default=None,
        description="Quantitative magnitude of the effect exactly as stated (e.g. '30% reduction in grain weight'), if given.",
    )
    evidence: str = Field(
        ...,
        description=(
            "Verbatim quote or near-verbatim paraphrase from the paper supporting this relation. "
            "Must cover both subject and object, not just one side of the relationship."
        ),
    )
    extraction_FLAG: Optional[bool] = Field(default=None, description="True if process extraction for Relation")


# -----------------------------
# Root schema
# -----------------------------

class ExtractionSchema(ExtractionBaseModel):
    metadata: Optional[Metadata] = Field(
        default=None,
        description="Paper metadata. Null to clear the entire section.",
    )
    entities: Optional[List[Entity]] = Field(
        default=None,
        description="All genes/proteins/phenotypes/tissues/substrates characterized in the paper. Null to clear; [] if confirmed none.",
    )
    relations: Optional[List[Relation]] = Field(
        default=None,
        description="All genotype-phenotype (or gene-gene, gene-substrate) relationships characterized in the paper. Null to clear; [] if confirmed none.",
    )
    patch_set: Optional[PatchSet] = Field(
        default=None,
        description="Only used for verification and updation process.",
    )


# -----------------------------
# Domain prompt configuration
# -----------------------------

ROLE = (
    "High-precision data extraction engine specialized in plant molecular genetics, "
    "focused on sugar transporter (SWEET family) genotype-phenotype mechanisms."
)

DOMAIN_NOTES = """\
- "SWEET" = Sugars Will Eventually be Exported Transporter gene/protein family: bidirectional,
  proton-independent sugar transporters, often acting at sink-source tissue interfaces.
- Use type="Gene" for the gene/allele itself; only create a separate type="Protein" entity if
  the paper discusses the protein product distinctly (e.g. structural or localization studies).
- relation_type="is_homolog_of" is only for explicit cross-gene/cross-species homology statements
  made by the paper (e.g. "ZmSWEET4c and OsSWEET4 are key players for..."), not mere co-membership
  in the SWEET family.
- expression_location captures tissue/cell/subcellular localization (e.g. "basal endosperm transfer
  layer (BETL)", "plasma membrane"), NOT chromosomal or genomic location.
- A single paper-reported phenotype (e.g. reduced grain weight, chlorosis) should be modeled as one
  Entity of type="Phenotype_Trait", linked to each responsible gene via its own Relation.
"""

EXTRACTION_PROMPT = ExtractionPrompt(role=ROLE, domain_notes=DOMAIN_NOTES)