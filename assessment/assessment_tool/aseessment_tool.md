# Assessment Tool Design

## 1) Goal

Build a lightweight human-review tool to evaluate LLM extraction quality across 7 experiment scenarios.  
The reviewer compares source markdown with extracted structured output, labels each extracted field, and saves annotations for later analysis.

## 2) Scope

### In scope
- Side-by-side review UI:
  - Left: source markdown content for one paper.
  - Right: extracted result JSON for one scenario.
- Per-field human annotation with fixed quality categories.
- Reviewer note per field (optional) and per paper-scenario (optional).
- Fast navigation:
  - Switch paper by paper id (for example DOI).
  - Switch scenario/case for the same paper.
- Persistent saving after review actions.
- Exportable annotation files (JSON/JSONL/CSV-ready).

### Out of scope (phase 1)
- Model reruns from UI.
- Automatic adjudication/conflict resolution.
- Multi-user real-time collaboration.

## 3) Input Data

- Source markdown directory: `assessment/samples/mineru`
- Extraction outputs: `assessment/runs/<case>/results/<paper_id>/merged_extraction.json` (or equivalent resolved path)

### Cases (7 total)
- `s1_gpt54_single`: GPT-5.4-mini single-path baseline
- `s2_gpt5_single`: GPT-5-mini single-path baseline
- `s3_gpt5_multi`: GPT-5-mini multi-path extraction
- `s4_gpt5_update1`: GPT-5-mini extraction + 1-turn update (from `s2_gpt5_single`)
- `s5_gpt5_update2`: GPT-5-mini extraction + 2-turn update (from `s4_gpt5_update1`)
- `s6_gpt5_verify`: GPT-5-mini extraction + verification (patches from `s2_gpt5_single`)
- `s7_gpt5_update_verify`: GPT-5-mini update + verification (patches from `s4_gpt5_update1`)

## 4) Annotation Taxonomy

Each extracted field gets one required label:

1. `correct`
2. `missing_critical_component`
3. `missing_minor_component`
4. `partial_wrong`
5. `missing_critical_component_and_partial_wrong`
6. `missing_minor_component_and_partial_wrong`
7. `completely_wrong`

Also store:
- `note` (free text, optional)
- `reviewed_at` (ISO timestamp)
- `reviewer_id` (string)

Default behavior:
- Storage default label is `correct` for every field unless reviewer changes it.
- UI shows an empty/unselected label state by default (no visible preselection).

## 5) Reviewer Workflow

1. Select paper id.
2. Select case.
3. Tool loads:
   - Source markdown for that paper.
   - Extraction JSON for that case and paper.
4. Reviewer annotates each target field.
5. Tool autosaves (or explicit save button) with confirmation.
6. Reviewer moves to next case or next paper.

## 6) UI Design (MVP)

### Layout
- Top bar:
  - Paper selector (searchable dropdown)
  - Case selector (7 fixed options)
  - Save status indicator
  - Progress indicator (for example `12/120 papers`, `4/7 cases`)
- Main split view:
  - Left pane: markdown viewer (rendered + raw toggle)
  - Right pane: extraction viewer + annotation controls

### Right-pane structure
- Collapsible list of extraction fields.
- For each field:
  - Field key/path
  - Extracted value
  - Label selector (7 categories)
  - Note textbox
- Global notes area for the full paper-case review.

### Usability details
- Keyboard shortcuts:
  - Number keys 1-7 assign labels.
  - `Ctrl/Cmd + S` saves.
  - Arrow keys move between fields.
- Label selector displays no initial choice, even though backend default is `correct`.
- Highlight unsaved changes.
- Warn before leaving with unsaved edits.

## 7) Data Model

Recommended storage format: one record per `(paper_id, case_id)`.

```json
{
  "paper_id": "10.1002_ird.164",
  "case_id": "s2_gpt5_single",
  "reviewer_id": "ma7",
  "reviewed_at": "2026-04-15T10:30:12Z",
  "global_note": "Most fields are usable; intervention details need correction.",
  "field_annotations": [
    {
      "field_path": "population.sample_size",
      "label": "missing_minor_component",
      "label_was_explicitly_set": true,
      "note": "Total N is present, subgroup N missing."
    },
    {
      "field_path": "intervention.duration",
      "label": "correct",
      "label_was_explicitly_set": false,
      "note": "Unit parsed as weeks but source says months."
    }
  ],
  "is_complete": true,
  "source_markdown_path": "assessment/samples/mineru/10.1002_ird.164.md",
  "extraction_path": "assessment/runs/s2_gpt5_single/results/10.1002_ird.164/merged_extraction.json",
  "tool_version": "v0.1.0"
}
```

Optional aggregate file for quick loading:
- `assessment/assessment_tool/data/annotations.jsonl`

Field-level schema notes:
- `label_was_explicitly_set=true`: reviewer manually selected the label in UI.
- `label_was_explicitly_set=false`: label was auto-filled as default `correct` during save.

## 8) File and Folder Proposal

```text
assessment/assessment_tool/
  aseessment_tool.md
  app/
    main.py                  # app entry (Streamlit/FastAPI+frontend)
    data_loader.py           # resolve paper/case file paths
    schema.py                # label enums and validation
    storage.py               # read/write annotations
    ui.py                    # page rendering and interactions
  data/
    annotations.jsonl
    review_state.json        # optional progress checkpoints
```

## 9) Validation and Error Handling

- If markdown is missing: show warning and disable submit.
- If extraction file is missing or malformed: show clear error state.
- Validate labels against enum before save.
- On save, fill any unselected field label with `correct` and set `label_was_explicitly_set=false`.
- If reviewer actively picks a label (including `correct`), set `label_was_explicitly_set=true`.
- Prevent duplicate rows by unique key:
  - `(paper_id, case_id, reviewer_id)` latest write wins.

## 10) Metrics to Track

- Coverage:
  - `% fields annotated per paper-case`
  - `% paper-cases completed`
- Quality distribution:
  - label counts by case
- Throughput:
  - reviewed fields/hour

## 11) Implementation Plan

### Phase 1 (MVP)
- Build local single-user app.
- Load markdown and extraction files from disk.
- Annotate per field and save one bundled record per `(paper_id, case_id)`.
- Provide paper/case switching and progress display.

### Phase 2
- Add filtering (only unreviewed, only problematic labels).
- Add comparison mode across 2 cases for same paper.
- Add CSV export and summary dashboard.

### Phase 3
- Add reviewer management and adjudication workflow.
- Add inter-rater agreement analysis support.

## 12) Recommended Tech (Practical)

- Fastest path: `Streamlit` app in Python.
- Alternative:
  - Backend: FastAPI
  - Frontend: React + split pane + virtualized field list

For current needs (single reviewer, rapid iteration), Streamlit is likely sufficient.
