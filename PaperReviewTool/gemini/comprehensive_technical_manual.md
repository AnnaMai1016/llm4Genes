# TileReview Information Extraction Pipeline: Exhaustive Technical Manual

This document provides a comprehensive, low-level technical reference for the TileReview information extraction system. It is designed to serve as the definitive guide for developers, covering every class, method, variable, and architectural decision in the codebase.

---

## Table of Contents
1. [Architectural Philosophy](#1-architectural-philosophy)
2. [Module: `src/gpt_extractor.py` (The Orchestrator)](#2-module-srcgpt_extractorpy-the-orchestrator)
    - 2.1 Class `PaperExtractor`
    - 2.2 Instance Variables
    - 2.3 Methods & Logic
3. [Module: `src/patches.py` (Atomic Corrections)](#3-module-srcpatchespy-atomic-corrections)
    - 3.1 `Patch` & `PatchSet` Models
    - 3.2 Patch Application Logic
4. [Module: `src/pydantic_util.py` (Structural Logic)](#4-module-srcpydantic_utilpy-structural-logic)
    - 4.1 `merge_results` Algorithm
    - 4.2 Schema Navigation & Normalization
5. [Module: `src/sentence_match.py` (Grounding Logic)](#5-module-srcsentence_matchpy-grounding-logic)
    - 5.1 Fuzzy Matching Algorithm
    - 5.2 Text Pre-processing
6. [Module: `src/normalizers.py` (Literal Mapping)](#6-module-srcnormalizerspy-literal-mapping)
7. [Module: `src/prompt.py` (Prompt Engineering)](#7-module-srcpromptpy-prompt-engineering)
8. [Module: `schema/meta_stru.py` (Data Models)](#8-module-schemameta_strupy-data-models)
9. [Module: `src/gpt_util.py` (API Utilities)](#9-module-srcgpt_utilpy-api-utilities)
10. [Module: `src/util.py` (General Helpers)](#10-module-srcutilpy-general-helpers)
11. [Integrated Usage Examples](#11-integrated-usage-examples)
12. [Standalone Module Usage](#12-standalone-module-usage)
13. [Logging & Snapshotting System](#13-logging--snapshotting-system)
14. [Error Handling & Troubleshooting](#14-error-handling--troubleshooting)
15. [Performance Tuning](#15-performance-tuning)

---

## 1. Architectural Philosophy

The TileReview pipeline is built on the principle of **Iterative Grounded Extraction**. 

### 1.1 Grounding
Every extracted data point must be traceable to a specific segment of the source text. This is enforced through the requirement of an `evidence` quote for every modification (Patch).

### 1.2 Iteration
The system does not rely on a single-pass extraction. Instead, it supports:
- **Focused Extraction**: Segmenting the schema into focus areas to manage token limits.
- **Verification**: A dedicated pass where the LLM audits its own work.
- **Correction**: Atomic updates (Patches) that are verified through fuzzy matching and optional secondary LLM checks.
- **Holistic Revision**: An `update` mode that re-evaluates the entire state given new context.

---

## 2. Module: `src/gpt_extractor.py` (The Orchestrator)

The `gpt_extractor.py` module defines the `PaperExtractor` class, which manages the state of a single paper's extraction session.

### 2.1 Class `PaperExtractor`

#### **Module-Level Symbols**
*   `_LOG_EXTRACTION_RUNS`: Filename for raw extraction API logs (`"extraction_runs.json"`).
*   `_LOG_EXTRACT_TURN_SNAPSHOTS`: Filename for snapshots after each merge pass.
*   `_LOG_VERIFICATION_RUNS`: Filename for verification API logs.
*   `_LOG_UNUSED_PATCHES`: Filename for pending patches.
*   `_LOG_PATCH_VERIFICATION_RESULTS`: Filename for patch audit results.

### 2.2 Instance Variables

| Variable | Type | Description |
|:---|:---|:---|
| `client` | `OpenAI` | The OpenAI API client. |
| `model` | `str` | The target LLM model (e.g., `"gpt-4o"`). |
| `file_unique_id` | `str` | A unique key for the paper, used in prompt caching. |
| `cache_time` | `str` | TTL for prompt cache (default `"24h"`). |
| `schema` | `Type[T]` | The Pydantic model class defining the data structure. |
| `document_text` | `str \| None` | The raw text of the paper. |
| `current_result` | `T \| None` | The current best-estimate extraction state. |
| `unused_patch_set` | `PatchSet` | A buffer for patches suggested but not yet applied. |
| `patch_verification_results` | `list[PatchVerification]` | Detailed audit of patch validity. |
| `extraction_runs` | `list[dict]` | Historical logs of all extraction API calls. |
| `extract_turn_snapshots` | `list[dict]` | Snapshots of `current_result` after merge operations. |
| `verification_runs` | `list[dict]` | Historical logs of all verification API calls. |
| `update_runs` | `list[dict]` | Historical logs of all holistic update calls. |
| `patch_application_runs` | `list[dict]` | Records of patch application events. |

### 2.3 Methods & Logic

#### **`set_document(document_text: str)`**
Stores the paper text. This is the prerequisite for all other operations.
*   **Example**: `extractor.set_document(full_markdown_text)`

#### **`extract_turn(NAME_FIELDS: list[list[str]] = None, verbosity="medium")`**
Orchestrates a multi-pass extraction. 
*   **Logic**: If `NAME_FIELDS` is provided, it iterates through groups (e.g., `['metadata']`, `['results']`), calls `_extract` for each, and merges the result using `merge_results`.
*   **Example**:
    ```python
    # Pass 1: Metadata. Pass 2: Results.
    extractor.extract_turn(NAME_FIELDS=[['metadata'], ['results']])
    ```

#### **`_extract(focus_area: list[str] = None, verbosity="medium") -> T`**
Internal method that makes the actual API call for extraction.
*   **Logic**: Calls `_response` with `verification=False`. Updates `self.current_result` and appends to `self.extraction_runs`.

#### **`verification(NAME_FIELDS: list[list[str]] = None, verbosity="medium")`**
Requests the LLM to identify errors in `current_result`.
*   **Logic**: Loops through focus areas, calls `_verification`, and populates `self.unused_patch_set`.
*   **Example**: `extractor.verification()`

#### **`update(NAME_FIELDS: list[list[str]] = None, verbosity="medium")`**
Performs a holistic re-evaluation.
*   **Logic**: Unlike verification (which uses patches), `update` asks the LLM to revise the model entirely. It overwrites `self.current_result` and clears the patch buffer.

#### **`verify_patches(min_match_score=40.0, verify_with_llm=True, ...)`**
Validates the `unused_patch_set`.
*   **Logic**:
    1.  Uses `sentence_match.extract_sentence_matches` to find the `evidence` quote in the paper.
    2.  If `verify_with_llm` is True, it performs a secondary API call (`_llm_verify_patch`) to confirm the semantic support for the change.
*   **Example**: `extractor.verify_patches(min_match_score=50.0)`

#### **`apply_verified_patches(strict_old_value=True, literal_normalizer=None, ...)`**
Commits verified patches.
*   **Logic**: Iterates through `patch_verification_results`. If `is_verified` is True, it applies the patch using `patches.apply_patches`.
*   **Example**: `extractor.apply_verified_patches(literal_normalizer=FuzzyNormalizer())`

#### **`save_logs_and_clear(...)`**
Flushes memory buffers to disk.
*   **Logic**: Appends to JSON arrays on disk using `_append_json_array`. Resets internal lists.

---

## 3. Module: `src/patches.py` (Atomic Corrections)

This module handles the logic for surgically modifying Pydantic models.

### 3.1 `Patch` & `PatchSet` Models

#### **`Patch` (BaseModel)**
*   `op`: `Literal["replace", "add", "append"]`.
*   `path`: JSON Pointer string (e.g., `/scale/site_count`).
*   `old_value`: The value expected at the path before modification.
*   `new_value`: The value to set or add.
*   `evidence`: The quote justifying the change.

#### **`PatchSet` (BaseModel)**
*   `patches`: A list of `Patch` objects.

### 3.2 Patch Application Logic

#### **`apply_patches(model: T, patches: list[Patch], strict_old_value: bool = True, literal_normalizer=None) -> T`**
Sequentially applies patches to a model.
*   **Logic**:
    1.  Converts the model to a plain dictionary.
    2.  Navigates the dictionary using JSON Pointer segments.
    3.  If `op == "replace"`, it verifies `old_value` if `strict_old_value` is True.
    4.  If `literal_normalizer` is provided, it attempts to normalize `new_value` labels.
    5.  Re-validates the final dictionary back into the model class.

#### **Example Standalone Use**
```python
from src.patches import Patch, apply_patches
from schema.meta_stru import ExtractionSchema

p = Patch(op="replace", path="/metadata/title", old_value="A", new_value="B", evidence="...")
updated_schema = apply_patches(current_schema_instance, [p])
```

---

## 4. Module: `src/pydantic_util.py` (Structural Logic)

Utilities for manipulating and navigating Pydantic models.

### 4.1 `merge_results` Algorithm

#### **`merge_results(a: T, b: T) -> T`**
Recursively merges two instances of the same model.
*   **Scalar Strategy**: `b` (the newer pass) overwrites `a`.
*   **List Strategy**: Concatenates lists.
*   **Model List Deduplication**: If a list contains Pydantic models, it deduplicates them based on their JSON-serialized representation.
*   **Example**: `final = merge_results(metadata_only, results_only)`

### 4.2 Schema Navigation & Normalization

#### **`literal_allowed_values_for_path(schema: type[BaseModel], path: str) -> tuple[str, ...] | None`**
Analyzes a Pydantic class to find the valid labels for a `Literal` field at a given path.
*   **Logic**: Walks the `model_fields` tree, unwrapping `Optional` and `Union` types to find `Literal` annotations.

#### **`normalize_literal_value_for_path(...)`**
Uses a normalizer to snap a value to the allowed labels for its path.
*   **Example**: `val = normalize_literal_value_for_path(Schema, "/relevance/level", "High", normalizer)`

---

## 5. Module: `src/sentence_match.py` (Grounding Logic)

Provides fuzzy string matching to verify LLM-provided evidence.

### 5.1 Fuzzy Matching Algorithm

#### **`extract_sentence_matches(context, target_sentence, window_size_padding=5, score_thr=40.0, context_words=50, max_results=3)`**
Finds the best matching segments in a large document.
*   **Logic**:
    1.  Cleans the document and target sentence of LaTeX, Markdown, and special characters.
    2.  Uses a sliding window approach to calculate `rapidfuzz.fuzz.ratio` scores.
    3.  Returns segments with their surrounding context (prefix/suffix).

### 5.2 Text Pre-processing
The `clean_for_matching` function ensures that formatting (like `\mathrm{...}` or `**bold**`) does not interfere with lexical matching.

---

## 6. Module: `src/normalizers.py` (Literal Mapping)

Defines strategies for mapping free-form LLM text to strict schema categories.

| Class | Strategy | Description |
|:---|:---|:---|
| `LiteralNormalizer` | ABC | Base class for all normalizers. |
| `FuzzyNormalizer` | Lexical | Uses `difflib.get_close_matches` (Levenshtein). |
| `EmbeddingNormalizer`| Semantic | Uses vector embeddings and cosine similarity. |
| `HybridNormalizer` | Mixed | Tries fuzzy first, falls back to embedding. |

*   **Example Standalone**:
    ```python
    norm = FuzzyNormalizer(cutoff=0.8)
    label = norm.normalize("Controlled-Drainage", ["Controlled Drainage", "No Drainage"])
    # label == "Controlled Drainage"
    ```

---

## 7. Module: `src/prompt.py` (Prompt Engineering)

Centralizes the instructions provided to the LLM.

### 7.1 `ExtractionPrompt` (Namespace)
*   `ROLE`: Defines the persona (Hydrology extraction expert).
*   `CORE_RULES`: Rules for grounding, nullability, and keyword selection.
*   `SYSTEM_PROMPT`: The standard system prompt for extraction calls.
*   `EXTRACTION_TASK`: Specific instructions for parsing new data.
*   `VERIFICATION_TASK`: Instructions for identifying errors and formatting patches.
*   `UPDATION_TASK`: Instructions for holistic revisions.

### 7.2 `build_user_prompt(...)`
Constructs the actual user message.
*   **Logic**: Wraps the paper text and conditionally appends previous extraction results based on the mode (`verification` vs `updation`).

---

## 8. Module: `schema/meta_stru.py` (Data Models)

Defines the structure of the agricultural drainage domain model.

### 8.1 Root Schema Components
The `ExtractionSchema` contains:
*   `metadata`: Location and paper identity.
*   `relevance`: Grounded relevance assessment.
*   `key_words`: Domain themes.
*   `context`: Objectives and scientific questions.
*   `scale`: Spatial and temporal resolution.
*   `methodology`: Approach types and models used.
*   `tile_design`: Drainage depth, spacing, and practices.
*   `results`: Summary bullets and quantitative statistics.

### 8.2 Nullability Convention
Every field is `Optional`. 
- `None` means the data is unknown or should be cleared.
- `[]` (empty list) means the data was checked and confirmed absent.

---

## 9. Module: `src/gpt_util.py` (API Utilities)

Low-level helpers for interacting with the OpenAI API.

*   **`usage_from_response(response)`**: Extracts token usage metadata, specifically breaking down `cached_input_tokens`.
*   **`response_meta(response)`**: Captures `response_id` and model version for reproducibility.

---

## 10. Module: `src/util.py` (General Helpers)

*   **`read_full_paper(filename)`**: Specialized reader for Mineru-generated Markdown. It truncates text at a stop-list (e.g., "References") and removes figure links to save tokens.
*   **`count_tokens(text, model_name)`**: Estimates token counts using `tiktoken`.

---

## 11. Integrated Usage Examples

### 11.1 Full Pipeline Flow
This example shows the standard multi-turn process.

```python
from openai import OpenAI
from src.gpt_extractor import PaperExtractor
from schema.meta_stru import ExtractionSchema, NAME_FIELDS

# 1. Setup
client = OpenAI()
extractor = PaperExtractor(schema=ExtractionSchema, client=client, model="gpt-4o")
extractor.set_document(open("paper.md").read())

# 2. Sequential Extraction
extractor.extract_turn(NAME_FIELDS=NAME_FIELDS)

# 3. Verification & Patch Generation
extractor.verification()

# 4. Patch Grounding & Verification
extractor.verify_patches(verify_with_llm=True)

# 5. Apply Verified Fixes
extractor.apply_verified_patches()

# 6. Persist Results
extractor.save_logs_and_clear(log_dir="./logs/run_001")
```

---

## 12. Standalone Module Usage

### 12.1 Manual Patching
You can use the patching logic independently of the LLM.

```python
from src.patches import Patch, apply_patches
from schema.meta_stru import ExtractionSchema

original = ExtractionSchema(...)
p = Patch(
    op="replace",
    path="/results/quantitative_stats/0/value",
    old_value="10",
    new_value="12",
    evidence="Table 2 explicitly states 12."
)

updated = apply_patches(original, [p], strict_old_value=True)
```

### 12.2 Standalone Evidence Search
Verify a quote in a document without an extractor.

```python
from src.sentence_match import extract_sentence_matches

results = extract_sentence_matches(
    context=paper_text,
    target_sentence="nitrate concentration was 5.2 mg/L",
    score_thr=60
)

for res in results:
    print(f"Found at {res['score']}% similarity: {res['highlight']}")
```

---

## 13. Logging & Snapshotting System

The pipeline implements an **Append-Only Serialized State** logging strategy.

### 13.1 `extraction_runs.json`
Stores every API transaction.
*   `run_index`: Sequential ID.
*   `focus_area`: The fields targeted.
*   `usage`: Token breakdown.
*   `parsed_result`: The raw output from the LLM.

### 13.2 Snapshots
*   `extract_turn_snapshots.json`: The merged state of `current_result` after each multi-pass group.
*   `unused_patch_set_patches.json`: Current pending corrections.

---

## 14. Error Handling & Troubleshooting

### 14.1 `PatchMismatchError`
**Cause**: `apply_verified_patches` failed because the `old_value` in the patch did not match the actual value in the model.
**Solution**: Disable `strict_old_value` (not recommended) or check if multiple patches are conflicting.

### 14.2 Low Grounding Scores
**Cause**: `verify_patches` is rejecting patches due to low fuzzy match scores.
**Solution**: 
1.  Lower `min_match_score` (e.g., to `30.0`).
2.  Enable `verify_with_llm` to let the model judge semantic similarity.

### 14.3 Token Limit Errors
**Cause**: The paper text + prompts exceed the context window.
**Solution**: Use a more granular `NAME_FIELDS` configuration (e.g., extract one top-level field at a time).

---

## 15. Performance Tuning

### 15.1 Prompt Caching
The pipeline uses `file_unique_id` to leverage OpenAI's prompt caching. Ensure this ID is consistent for the same paper across turns.

### 15.2 Fuzzy Score Thresholds
The `min_match_score` in `verify_patches` is the most sensitive parameter. 
- **Higher (70+)**: High precision, but may miss paraphrased evidence.
- **Lower (40-)**: High recall, but may accept irrelevant text segments.

---

*Manual End. Version 1.1.0 (2026-04-13).*
