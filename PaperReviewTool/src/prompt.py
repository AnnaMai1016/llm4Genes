import json
from pydantic import BaseModel
from typing import TypeVar
from .patches import Patch
ModelType = TypeVar('ModelType', bound=BaseModel)

GENERIC_CORE_RULES = """CRITICAL RULES:

1. STRICT GROUNDING:
- Use ONLY information explicitly stated in the paper.
- If information is not explicitly stated → return null.
- Do NOT infer, assume, generalize, or use external knowledge.

2. FIDELITY:
- Preserve exact numerical values and units.
- Preserve author wording for qualitative findings when possible.

3. STRUCTURE:
- Output must strictly follow the provided schema.
- Do NOT add extra fields.
- Do NOT change field names or hierarchy.

4. KEYWORDS:
- Extract/Summarize high-level scientific themes only.
- Do NOT include:
  - numbers
  - specific chemicals
  - local/place names
"""


class ExtractionPrompt:
    """
    Domain-specific system-prompt configuration.

    `role` and `domain_notes` carry everything that changes between domains (e.g.
    tile-drainage hydrology vs. SWEET-family genotype-phenotype mechanisms).
    `EXTRACTION_TASK` / `VERIFICATION_TASK` / `UPDATION_TASK` below stay domain-agnostic
    and are shared by every instance.

    Instantiate one of these per domain next to that domain's schema (see
    `schema/meta_stru.py` or `schema/sweet_transporter.py`) and pass it to
    `PaperExtractor(prompt=...)`.
    """

    def __init__(self, role: str, domain_notes: str = ""):
        self.role = role
        self.domain_notes = domain_notes.strip()

    @property
    def system_prompt(self) -> str:
        domain_block = f"DOMAIN CONTEXT:\n{self.domain_notes}\n\n" if self.domain_notes else ""
        return f"""
{self.role}

You extract or verify structured information from scientific papers.

{domain_block}{GENERIC_CORE_RULES}
"""

    # -----------------------------
    # EXTRACTION TASK
    # -----------------------------
    EXTRACTION_TASK = """
TASK:

Extract structured information from the paper.

- Populate ONLY fields supported by explicit text.
- Leave missing fields as null or empty.
- Do NOT hallucinate.
"""
        
    # -----------------------------
    # VERIFICATION TASK (PATCH MODE)
    # -----------------------------
    VERIFICATION_TASK = """
TASK: Verification (Patch Mode)

You are given:
1. The original paper text
2. A previous extraction result

Your job is to produce a PatchSet that corrects errors ONLY.

--------------------------------------------------
CORE PRINCIPLE:
Verification = minimal correction under strict grounding.
NOT rewriting, NOT enrichment, NOT summarization.
--------------------------------------------------

INSTRUCTIONS:

1. OUTPUT:
- ONLY return patch_set
- Keep other fields empty

2. ATOMICITY:
- Each patch must modify exactly ONE field
- Each patch must address ONE issue only

3. WHEN TO PATCH (STRICT GATE):
You may create a patch ONLY if:
A. The current value is explicitly contradicted by the source text
OR
B. The field is required and missing, AND the information is explicitly present

If BOTH conditions are false → DO NOT PATCH

4. FORBIDDEN ACTIONS (CRITICAL):
- Do NOT merge multiple findings into one field
- Do NOT increase level of detail
- Do NOT append additional clauses to an already correct statement
- Do NOT rephrase or improve wording if meaning is already correct
- Do NOT replace a correct value with a more detailed or “better” version
- Do NOT change null unless explicit information exists

5. SCOPE PRESERVATION:
- Each field must retain its original semantic scope
- Do NOT introduce new variables, conditions, comparisons, or time ranges

6. PATCH OPERATIONS:
- "replace" → ONLY if the original value is incorrect (path includes index for list elements).
- "add" → add a missing object key, insert at a list index, append to list using "-" as index, or use JSON Pointer "add" semantics as implemented by the app.
- "append" → ONLY for list fields: path must end at the list *field* (e.g. "/results/summary_bullets"), NOT at an index. new_value is one item or a list of items to append when the list exists but is missing entries supported by the source.

7. MINIMALITY:
- Changes must be the smallest possible edit
- Do not expand sentence length unnecessarily
- Avoid adding secondary clauses

8. EVIDENCE (MANDATORY):
- Each patch MUST include a direct quote or near-verbatim support
- Evidence must directly justify the change (not just be related)

9. CONFIDENCE:
- Only include patches with high confidence
- If uncertain → DO NOT PATCH

10. DEFAULT ACTION:
- If the existing value is supported by the source → DO NOTHING
"""
    # -----------------------------
    # UPDATE TASK
    # -----------------------------
    UPDATION_TASK = """
TASK: Verify and update a previous extraction result using the original paper text.

You are given:
1. The original paper text
2. A previous extraction result

INSTRUCTIONS:

1. OUTPUT: Return the updated extraction result and document all changes in patch_set.

2. CHANGE ELIGIBILITY:
Make changes and create a patch ONLY IF at least one condition is true:
  A. The current value is explicitly contradicted by the source text, OR
  B. The field is missing AND the information is explicitly present in the source.
If neither condition is met → DO NOT CHANGE and PATCH. Default action is to do nothing.

3. UPDATE OPERATIONS:
  "replace" → current value is factually incorrect
  "add"     → field is entirely absent or insert at a specific list index
  "append"  → path is the list field; append new_value (one item or list of items) to that list when items are missing and supported by the source

4. RULES (ALL MANDATORY):
ATOMICITY
  Each patch modifies exactly ONE field
  Each patch addresses exactly ONE issue

MINIMALITY
  Make the smallest possible edit
  Do not expand sentence length or add secondary clauses

SCOPE PRESERVATION
  Retain the original semantic scope of each field
  Do not introduce new variables, conditions, comparisons, or time ranges

EVIDENCE (required per patch)
  Include a direct quote or near-verbatim excerpt from the source
  Evidence must directly justify the change, not merely relate to it

CONFIDENCE
  Only patch when confidence is high
  If uncertain → DO NOT PATCH

FORBIDDEN ACTIONS
  Do NOT merging multiple findings into one field
  Do NOT increase level of detail beyond what the source states
  Do NOT append clauses to an already-correct statement
  Do NOT rephrase or reword if the meaning is already correct
  Do NOT replace a correct value with a more detailed or "better" version
  Do NOT change null unless explicit information exists in the source
"""

# -----------------------------
# USER PROMPT BUILDER
# -----------------------------

def build_user_prompt(
    full_text: str,
    focus_area: list[str] | None = None,
    verification: bool = False,
    updation: bool = False,
    previous_extraction: ModelType | None = None,
) -> str:

    prefix = (
        "Paper content follows. Treat this as the ONLY source of truth.\n\n"
        "=== PAPER START ===\n"
        f"{full_text}\n"
        "=== PAPER END ===\n"
    )

    if updation:
        prompt = prefix + "\n\n" 
        
        prompt += "PREVIOUS EXTRACTION:\n"
        if previous_extraction is None:
            raise ValueError("previous_extraction is required when updation=True.")
        prompt += json.dumps(previous_extraction.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        prompt += "\n\n"
        prompt += ExtractionPrompt.UPDATION_TASK

        if focus_area:
            prompt += (
                "\n\nFOCUS CONSTRAINT:\n"
                f"Only update the following sections: {', '.join(focus_area)}.\n"
                "Keep other fields unchanged.\n"
            )
        return prompt

    if verification:
        prompt = prefix + "\n\n"

        prompt += "PREVIOUS EXTRACTION:\n"
        if previous_extraction is None:
            raise ValueError("previous_extraction is required when verification=True.")
        prompt += json.dumps(previous_extraction.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        prompt += "\n\n"

        prompt += ExtractionPrompt.VERIFICATION_TASK

        if focus_area:
            prompt += (
                "\n\nFOCUS CONSTRAINT:\n"
                f"Only verify and patch the following sections: {', '.join(focus_area)}.\n"
                "Ignore all other fields.\n"
                "If above session are missing in the *PREVIOUS EXTRACTION*, skip the missed fields."
            )
        return prompt
    
    
    prompt = prefix + "\n\n" + ExtractionPrompt.EXTRACTION_TASK

    if focus_area:
        prompt += (
            "\n\nFOCUS CONSTRAINT:\n"
            f"Only extract the following sections: {', '.join(focus_area)}.\n"
            "Leave all other fields empty."
        )
    else:
        prompt += (
            "\n\nExtract all fields defined in the schema "
            "(except patch_set)."
        )

    return prompt

class ValidationPrompt():
    ROLE = "You are a precision fact-checker for structured data extraction."
    TASK = "Your task is to verify whether a proposed change to extracted data is valid given the source text evidence."
    CORE_RULES = """
1. The change is VALID only if the new value is explicitly supported by ANY of the matched contexts
2. The change is INVALID if:
   - None of the contexts clearly support the new value
   - The evidence is ambiguous or could be interpreted differently
   - The old value is already correct based on the contexts
3. Be conservative: when in doubt, mark as INVALID
4. Consider the field's intended meaning based on its description
5. Review ALL provided contexts - the best match isn't always first
6. IMPORTANT: Focus on semantic/content validity, not textual similarity
   - The evidence may be paraphrased by the LLM
   - Low fuzzy match scores DO NOT automatically invalidate a change
   - Small word changes can completely alter meaning - judge semantically
   - A lower-scored match may still provide valid semantic support"""

    SYSTEM_PROMPT = f"{ROLE}\n"
    SYSTEM_PROMPT += f"{TASK}\n"
    SYSTEM_PROMPT += f"CRITICAL RULES:\n{CORE_RULES}"

def llm_verify_user_prompt(
    patch: Patch,
    field_desc_section: str,
    contexts_section: str
) -> str:
    """Build the user prompt for verifying a patch"""

    user_prompt = f"""PROPOSED CHANGE:
- Field path: {patch.path}
- Operation: {patch.op}
- Old value: {patch.old_value}
- New value: {patch.new_value}
- Evidence provided: "{patch.evidence}"
{field_desc_section}{contexts_section}

QUESTION: Is the proposed change valid given the matched contexts above?

Consider:
1. Does ANY of the matched contexts semantically support the new value?
2. Is the change necessary, or is the old value already correct?
3. Does the new value align with the field's intended meaning (see field description)?
4. Focus on meaning and content, not just textual similarity to the evidence

NOTE: Fuzzy match scores are provided for reference only. A lower score doesn't mean the context is invalid - the evidence may be paraphrased. Judge based on semantic content.

Provide your verification decision."""
    return user_prompt

"""
- Do NOT add new facts that were not in the original field
"""