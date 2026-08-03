# Extraction Assessment Plan (Revised)

## 1) Goal

Evaluate extraction quality and cost-efficiency before full-scale deployment by comparing model settings and extraction strategies on a fixed sample of 50 randomly selected papers.

## 2) Scope

- **Sample size**: 50 papers (randomly sampled, fixed seed for reproducibility).
- **Target schema**: all extraction fields currently used in the review pipeline.
- **Primary model comparison**:
  - `GPT-5.4-mini`
  - `GPT-5-mini`

## 3) Strategies to Compare

### A. Extraction Only

1. **Single-path extraction**  
   Extract all fields in one pass.

2. **Multi-path extraction**  
   Split extraction into multiple passes (example 5-path layout):

```python
NAME_FIELDS = [
    ["metadata", "relevance"],
    ["context"],
    ["scale", "methodology", "tile_design"],
    ["results", "internal_evaluation"],
    ["key_words", "overall_synthesis"],
]
```

### B. Extraction + Update

- Start from single-path extraction output.
- Run 1-turn and 2-turn update variants.
- The 2-turn update uses the 1-turn updated result as input.

### C. Extraction + Verification

- Start from single-path extraction output.
- Run 1 verification turn.
- Apply verification patches to produce final output.

### D. Extraction + Update + Verification

- Start from the 1-turn updated output (from Strategy B).
- Run 1 verification turn.
- Apply verification patches to produce final output.

## 4) Experiment Matrix

- `GPT-5.4-mini`:
  - Run **Extraction Only: Single-path** (baseline due to cost).
- `GPT-5-mini`:
  - Run all strategies: A (single + multi), B, C, D.

This setup balances quality analysis depth and budget control.

## 5) Evaluation Criteria

For each strategy/model setting, assess:

1. **Field-level accuracy**
   - Correctness against reference labels (or adjudicated gold subset).
2. **Completeness**
   - Missing-field rate.
3. **Consistency**
   - Internal coherence across related fields.
4. **Edit distance to accepted output**
   - Number of human corrections required (or patch count).
5. **Cost and latency**
   - Tokens, API cost, and end-to-end runtime.

## 6) Execution Workflow

1. Generate and freeze the 50-paper sample list.
2. Run baseline single-path extraction for both models.
3. For `GPT-5-mini`, execute derived strategies sequentially using prior outputs:
   - Update uses single-path results.
   - Verification uses single-path or updated results (depending on strategy).
4. Use Batch API whenever possible for extraction and verification stages.
5. Store all intermediate outputs and patches for reproducibility.
6. Score all runs with the same evaluation script/rubric.

## 7) Cost-Control Rules

- Prefer **Batch API** for large, independent jobs.
- Reuse prior outputs for update/verification instead of rerunning extraction.
- Limit `GPT-5.4-mini` to baseline single-path extraction.
- Perform offline patch application and metric computation.

## 8) Deliverables

- A comparison table by strategy/model:
  - Accuracy, completeness, consistency, correction effort, cost, latency.
- Error analysis by field category.
- Final recommendation for the production extraction workflow.

## 9) Decision Rule

Select the strategy that provides the best quality-cost tradeoff, with priority order:

1. Accuracy and completeness
2. Correction effort
3. Cost and runtime

If two strategies are close in quality, choose the lower-cost and simpler pipeline.

## 10) Execution Checklist

- [X] Fix and record random seed for sampling.
- [ ] Generate the 50-paper sample list and freeze it.
- [ ] Prepare output directories for each model/strategy.
- [ ] Run `GPT-5.4-mini` single-path baseline.
- [ ] Run `GPT-5-mini` single-path extraction.
- [ ] Run `GPT-5-mini` multi-path extraction.
- [ ] Run `GPT-5-mini` extraction + 1-turn update.
- [ ] Run `GPT-5-mini` extraction + 2-turn update.
- [ ] Run `GPT-5-mini` extraction + verification.
- [ ] Run `GPT-5-mini` extraction + update + verification.
- [ ] Apply all verification patches offline and save final JSON outputs.
- [ ] Compute metrics for all runs using the same scoring script.
- [ ] Build comparison table (quality, correction effort, cost, runtime).
- [ ] Complete error analysis by field category.
- [ ] Write final recommendation with selected production strategy.
