# TileReview Information Extraction Pipeline: Exhaustive Technical Manual (v1.2)

This document provides a comprehensive, low-level technical reference for the TileReview information extraction system. It is designed to serve as the definitive guide for developers, covering every class, method, variable, and architectural decision in the codebase.

---

## Table of Contents
1. [Architectural Philosophy & Data Flow](#1-architectural-philosophy--data-flow)
2. [Module: `src/gpt_extractor.py` (The Orchestrator)](#2-module-srcgptextractorpy-the-orchestrator)
    - 2.1 Class `PaperExtractor`
    - 2.2 Instance Variables (Detailed)
    - 2.3 Core Public Methods & Examples
    - 2.4 Internal Logic & Private Methods
3. [Module: `src/patches.py` (Atomic Corrections)](#3-module-srcpatchespy-atomic-corrections)
    - 3.1 `Patch` & `PatchSet` Models
    - 3.2 Verification-Related Models
    - 3.3 Patch Application Logic (`apply_patches`)
4. [Module: `src/pydantic_util.py` (Structural Logic)](#4-module-srcpydantic_utilpy-structural-logic)
    - 4.1 Deep Merging (`merge_results`)
    - 4.2 Schema Navigation & Normalization
5. [Module: `src/sentence_match.py` (Grounding Logic)](#5-module-srcsentence_matchpy-grounding-logic)
    - 5.1 Fuzzy Matching Algorithm
    - 5.2 Text Pre-processing
6. [Module: `src/normalizers.py` (Literal Mapping)](#6-module-srcnormalizerspy-literal-mapping)
7. [Module: `src/prompt.py` (Prompt Engineering)](#7-module-srcpromptpy-prompt-engineering)
8. [Module: `schema/meta_stru.py` (Data Models)](#8-module-schemameta_strupy-data-models)
9. [Module: `src/gpt_util.py` (API Utilities)](#9-module-srcgpt_utilpy-api-utilities)
10. [Module: `src/util.py` (General Helpers)](#10-module-srcutilpy-general-helpers)
11. [Step-by-Step Integrated Workflow](#11-step-by-step-integrated-workflow)
12. [Standalone Module Usage](#12-standalone-module-usage)
13. [The Logging & Snapshotting System](#13-the-logging--snapshotting-system)
14. [Error Handling & Troubleshooting](#14-error-handling--troubleshooting)
15. [Performance Tuning](#15-performance-tuning)

---

## 1. Architectural Philosophy & Data Flow

The TileReview pipeline is built on the principle of **Iterative Grounded Extraction**. This architecture addresses the core challenges of using LLMs for structured data extraction: reliability, auditability, and context management.

### 1.1 Grounding
Every extracted data point must be traceable to a specific segment of the source text. This is enforced through the `evidence` field required for every `Patch`, which is a verbatim or near-verbatim quote from the paper. This quote is then programmatically verified against the document using fuzzy string matching.

### 1.2 Iteration & Refinement
The system uses a multi-turn, stateful conversation with the LLM. There are three main modes of operation:
- **`extract`**: The initial pass to generate a structured Pydantic object from raw text.
- **`verification`**: An audit pass where the LLM reviews its previous work (`current_result`) and proposes corrections as atomic `Patch` objects. This does **not** change the main result.
- **`update`**: A holistic revision pass where the LLM is asked to re-write the entire extraction based on the previous version, resolving contradictions directly.

The standard workflow is `extract` -> `verification` -> `verify_patches` -> `apply_verified_patches`.

### 1.3 Data Flow Diagram
```text
┌────────────────┐     ┌───────────────────┐     ┌────────────────┐
│  Source Text   │────▶│ PaperExtractor    │────▶│ current_result │
│ (paper.md)     │     │ (State Machine)   │     │ (Pydantic Model) |
└────────────────┘     └─────────┬─────────┘     └────────┬───────┘
                                 │                        │
     (Verification Turn) ◀───────┘                        │ (Apply Patches)
                                 │                        │
                         ┌───────▼──────────┐     ┌───────▼──────────┐
                         │ unused_patch_set │────▶│ apply_patches()  │
                         │ (Staged Changes) │     │ (Atomic Updates) │
                         └──────────────────┘     └──────────────────┘
```

---

## 2. Module: `src/gpt_extractor.py` (The Orchestrator)

This module defines the `PaperExtractor` class, the stateful orchestrator of the entire pipeline.

### 2.1 Class `PaperExtractor`

#### **Module-Level Constants**
*   **`_LOG_*` variables**: A series of string constants defining the filenames for the append-only JSON log files (e.g., `_LOG_EXTRACTION_RUNS = "extraction_runs.json"`).
*   **`_SNAPSHOT_*` variables**: Filenames for single-file JSON snapshots written by `save_current_patch_artifacts` (e.g., `_SNAPSHOT_CURRENT_RESULT = "current_result.json"`).

### 2.2 Instance Variables (Detailed)

| Variable | Type | Description |
|:---|:---|:---|
| `client` | `OpenAI` | The instantiated OpenAI API client. |
| `model` | `str` | The target LLM model identifier (e.g., `"gpt-4o"`). |
| `file_unique_id` | `str` | A unique key for the document, critical for enabling OpenAI's prompt caching feature. |
| `cache_time` | `str` | The Time-To-Live for the prompt cache (default `"24h"`). |
| `schema` | `Type[T]` | The Pydantic model class that defines the expected structured output. |
| `document_text` | `str \| None` | The raw text of the paper, set via `set_document()`. |
| `current_result` | `T \| None` | The working Pydantic model instance holding the current state of the extraction. This is the object that is refined over time. |
| `unused_patch_set` | `PatchSet` | A buffer holding a list of `Patch` objects suggested during a `verification` turn but not yet verified or applied. |
| `patch_verification_results`| `list[PatchVerification]` | Detailed results from the last `verify_patches` call, aligning with `unused_patch_set`. |
| `extraction_runs` | `list[dict]` | A low-level, in-memory log of every extraction API call, including tokens, response IDs, and the raw parsed output. |
| `extract_turn_snapshots`| `list[dict]` | A high-level log of the merged state of `current_result` after each complete `extract_turn`. |
| `verification_runs` | `list[dict]`| A log of every verification API call that generated patches. |
| `update_runs` | `list[dict]` | A log of every holistic revision (`update`) API call. |
| `patch_application_runs`| `list[dict]`| An audit trail of every time `apply_verified_patches` was run, detailing which patches were applied or skipped. |
| `_patch_artifacts_dir`| `Path` | The directory where live snapshots are automatically saved. |

### 2.3 Methods & Logic

#### `set_document(document_text: str)`
*   **Description**: Ingests the paper's text content into the extractor's state. This is the mandatory first step.
*   **Example**: `extractor.set_document(open("my_paper.md").read())`

#### `extract_turn(NAME_FIELDS=None, verbosity="medium")`
*   **Description**: Orchestrates a full extraction pass. It can process the document in one go or in sequential chunks (`focus_area`) to manage token limits.
*   **Internal Logic**: If `NAME_FIELDS` is provided, it loops through each list of fields, calls `_extract` for that group, and then merges the temporary result into `self.current_result` using `pydantic_util.merge_results`.
*   **Example**:
    ```python
    from schema.meta_stru import NAME_FIELDS
    # Use pre-defined field groups to extract the entire paper section by section
    extractor.extract_turn(NAME_FIELDS=NAME_FIELDS)
    ```

#### `verification(NAME_FIELDS=None, verbosity="medium", ...)`
*   **Description**: An audit pass. It asks the LLM to review the `current_result` and propose minimal corrections in the form of `Patch` objects.
*   **Internal Logic**: Calls `_response` with `verification=True`. The returned `PatchSet` is deduplicated and its patches are appended to `self.unused_patch_set`.
*   **Example**: `extractor.verification()`

#### `update(NAME_FIELDS=None, verbosity="medium")`
*   **Description**: A holistic revision pass. It asks the LLM to re-write the extraction, providing the previous version as context. This is more powerful but less surgical than patching.
*   **Internal Logic**: Calls `_response` with `updation=True`. The result directly overwrites `self.current_result` and clears the patch buffer.

#### `verify_patches(min_match_score=40.0, verify_with_llm=True, ...)`
*   **Description**: Validates the `evidence` for each patch in `unused_patch_set`.
*   **Internal Logic**: For each patch, it calls `_verify_single_patch`, which uses `sentence_match.extract_sentence_matches` to find the quote in the text. If `verify_with_llm` is `True`, it makes a second, cheaper API call to a "Fact-Checker" persona to semantically validate the change.
*   **Example**: `extractor.verify_patches(min_match_score=50.0)`

#### `apply_verified_patches(strict_old_value=True, literal_normalizer=None, ...)`
*   **Description**: Commits verified patches to `current_result`.
*   **Internal Logic**: Filters `patch_verification_results` for `is_verified=True`. If a `literal_normalizer` is provided, it snaps patch values to valid schema `Literal`s. Finally, it calls `patches.apply_patches` to update the model.
*   **Example**:
    ```python
    from src.normalizers import FuzzyNormalizer
    extractor.apply_verified_patches(literal_normalizer=FuzzyNormalizer(cutoff=0.8))
    ```

### 2.4 Internal Logic & Private Methods

*   **`_response(...)`**: The central function for making all OpenAI API calls. It constructs the user prompt via `build_user_prompt` and calls `client.responses.parse`.
*   **`_verify_single_patch(...)`**: The core of the grounding logic. It orchestrates the fuzzy match and optional LLM fact-check for a single patch.
*   **`_llm_verify_patch(...)`**: Builds the prompt for the Fact-Checker LLM and parses its `VerificationResult`.

---

## 3. Module: `src/patches.py` (Atomic Corrections)

### 3.1 `Patch` & `PatchSet` Models
- **`Patch`**: Models a single change with `op`, `path`, `old_value`, `new_value`, and `evidence`.
- **`PatchSet`**: A simple container for a list of `Patch` objects.

### 3.2 Verification-Related Models
- **`VerificationResult`**: The Pydantic model for the Fact-Checker LLM's response (`is_valid`, `reasoning`).
- **`PatchVerification`**: An aggregate model that bundles a `Patch`, its `matched_contexts` from the document, and the `VerificationResult`.

### 3.3 Patch Application Logic (`apply_patches`)
*   **`_pointer_segments(path)`**: Parses a JSON Pointer string into a list of keys/indices.
*   **`_get_node(data, segments)`**: Traverses a nested dictionary/list structure.
*   **`_apply_patch_to_dict(data, patch, ...)`**: The in-place modification function for a plain dictionary. It handles `replace`, `add` (including list insertion), and `append` logic.
*   **`apply_patches(...)`**: The public-facing function. It dumps the Pydantic model to a dict, sequentially calls `_apply_patch_to_dict` for each patch, and then re-validates the final dict into a new Pydantic model instance.

---

## 4. Module: `src/pydantic_util.py` (Structural Logic)

### 4.1 Deep Merging (`merge_results`)
*   **`_merge_values(...)`**: The recursive heart of the merge. It dispatches to `_merge_lists` or `_merge_scalarish` based on type.
*   **`_merge_lists(...)`**: Handles list merging. It concatenates lists and, if they contain Pydantic models, deduplicates them by their serialized JSON representation to prevent identical entries from accumulating across turns.

### 4.2 Schema Navigation & Normalization
*   **`get_field_description(path, schema)`**: Retrieves the docstring for a field from the schema.
*   **`literal_allowed_values_for_path(schema, path)`**: Returns the valid `Literal` choices for a given field path.
*   **`normalize_literal_value_for_path(...)`**: The bridge between a patch's `new_value` and the schema's constraints.

---

## 5. Module: `src/sentence_match.py` (Grounding Logic)

### 5.1 Fuzzy Matching Algorithm (`extract_sentence_matches`)
1.  **Cleaning**: `clean_for_matching` removes LaTeX, Markdown, and special characters.
2.  **Tokenization**: The document is split into tokens while preserving whitespace.
3.  **Sliding Window**: A window of tokens slides across the document.
4.  **Scoring**: `rapidfuzz.fuzz.ratio` compares the window text to the target evidence.
5.  **Ranking & Filtering**: Matches are sorted by score, and overlapping results are removed.

---

## 6. Module: `src/normalizers.py` (Literal Mapping)

*   **`FuzzyNormalizer`**: Uses `difflib.get_close_matches` for lexical similarity. Good for typos.
*   **`EmbeddingNormalizer`**: Uses `sentence-transformers` and cosine similarity for semantic mapping. More powerful but slower.
*   **`HybridNormalizer`**: Tries fuzzy first, then embedding as a fallback.

---

## 7. Module: `src/prompt.py` (Prompt Engineering)

This module defines the "brains" of the LLM.
*   **`ExtractionPrompt`**: Contains the system prompts and task descriptions for `extract`, `verification`, and `update`.
*   **`ValidationPrompt`**: Defines the persona for the cheaper, faster "Fact-Checker" LLM used in `_llm_verify_patch`.

---

## 8. Module: `schema/meta_stru.py` (Data Models)

Defines the target data structure.
*   **`ExtractionBaseModel`**: Sets `extra="forbid"` to prevent the LLM from inventing new fields.
*   **`ExtractionSchema`**: The root model.
*   **Sub-Models**: `Relevance`, `Location`, `Metadata`, `Context`, `Scale`, `Methodology`, `TileDesign`, `Results`, `InternalEvaluation`, `OverallSynthesis`. Each field has a detailed `description` to guide the LLM.
*   **`NAME_FIELDS`**: Pre-defined focus groups for `extract_turn`.

---

## 9. Module: `src/gpt_util.py` (API Utilities)

*   **`usage_from_response(response)`**: Parses the `usage` object from the OpenAI response, calculating cached vs. non-cached tokens.
*   **`response_meta(response)`**: Extracts the `id` and `model` from the response for logging.

---

## 10. Module: `src/util.py` (General Helpers)

*   **`read_full_paper(filename)`**: A specialized file reader that intelligently truncates papers at the "References" section to save tokens.
*   **`count_tokens(text, ...)`**: Uses `tiktoken` to estimate prompt size.

---

## 11. Step-by-Step Integrated Workflow

This is the canonical end-to-end process.

```python
from openai import OpenAI
from src.gpt_extractor import PaperExtractor
from schema.meta_stru import ExtractionSchema, NAME_FIELDS
from src.normalizers import FuzzyNormalizer

# 1. Initialize
client = OpenAI()
extractor = PaperExtractor(schema=ExtractionSchema, client=client, model="gpt-4o", file_unique_id="paper123")
extractor.set_document(open("paper.md").read())

# 2. Initial Extraction
print("Starting initial extraction...")
extractor.extract_turn(NAME_FIELDS=NAME_FIELDS)
print(f"Extracted title: {extractor.current_result.metadata.title}")

# 3. Verification / Audit
print("Starting verification pass...")
extractor.verification()
print(f"Found {len(extractor.unused_patch_set.patches)} potential corrections.")

# 4. Grounding & Validation
print("Validating patches against source text...")
extractor.verify_patches(verify_with_llm=True)

# 5. Apply & Finalize
print("Applying verified patches...")
extractor.apply_verified_patches(literal_normalizer=FuzzyNormalizer())

# 6. Save & Persist
print("Saving session logs...")
extractor.save_logs_and_clear(log_dir="./logs/paper123_run1")
print("Done.")
```

---

## 12. Standalone Module Usage

#### **Manual Patching**
```python
from src.patches import Patch, apply_patches
# Assuming 'model_instance' is an instance of ExtractionSchema
manual_patch = Patch(op="replace", path="/scale/site_count", old_value=2, new_value=4, evidence="The study used four sites...")
corrected_model = apply_patches(model_instance, [manual_patch], strict_old_value=True)
```

#### **Standalone Evidence Search**
```python
from src.sentence_match import extract_sentence_matches
matches = extract_sentence_matches(paper_text, "nitrate loads were reduced by 50%", score_thr=60)
```

---

## 13. The Logging & Snapshotting System

The pipeline produces two types of logs:
1.  **Append-Only Logs**: JSON arrays where new data is always appended. This is the primary audit trail.
2.  **Snapshots**: Single JSON files that are overwritten on each change for live monitoring.

| File | Type | Description |
|:---|:---|:---|
| `extraction_runs.json` | Append-Only | Every extraction API call. |
| `extract_turn_snapshots.json` | Append-Only | The merged result after a full turn. |
| `patch_application_runs.json`| Append-Only | Record of which patches were applied. |
| `current_result.json` | Snapshot | The live state of `current_result`. |
| `unused_patch_set_patches.json`| Snapshot | The live state of pending patches. |

---

## 14. Error Handling & Troubleshooting

*   **`PatchMismatchError`**: `apply_verified_patches` failed because the `old_value` was stale. Re-run `verification()` and `verify_patches()`.
*   **`TokenLimitError`**: The document is too long. Use a more granular `NAME_FIELDS` in `extract_turn`.
*   **Low Grounding Scores**: Lower `min_match_score` in `verify_patches` or improve the `evidence` quality in your prompts.

---

## 15. Performance Tuning

*   **`NAME_FIELDS` Granularity**: Fine-tuning the focus groups in `extract_turn` is the primary way to balance cost, speed, and accuracy.
*   **`min_match_score`**: Adjust this threshold in `verify_patches` to control the precision/recall of the grounding check.
*   **LLM Choice**: Use a faster/cheaper model for `verification` turns if patch generation is not complex. The main `PaperExtractor` class can be subclassed to override the `_llm_verify_patch` method to use a different model for fact-checking.

---
*Manual End. Version 1.2.0 (2026-04-13).*
