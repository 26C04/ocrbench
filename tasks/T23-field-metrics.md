# T23: Field metrics and layer attribution

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §4, §5, §7 |
| Depends on | T20 |
| Followed by | T24 |
| Main outputs | `src/ocrbench/field_metrics.py` (reuses the statistics in `metrics.py`), tests |

## Goal

Compute the ADR-0002 metrics from per-sheet results.

## Implementation

1. Per sheet: correct / wrong / null for each field; silent error = non-null and wrong.
2. Run summary: field accuracy (primary), sheets fully correct with Wilson interval,
   per-field accuracy with fixed-seed bootstrap, silent errors, abstentions,
   time p50/p95, cost per sheet.
3. Paired comparison of two runs on the same sheets: reuse `mcnemar_exact`.
4. Layer attribution for pipelines with `layer1_text`: NFKC + whitespace normalize both
   sides; correct value absent → Layer 1, present → Layer 2.
5. Decision-rule helper (ADR-0002 §7): given the best local run and the Mistral run,
   return `local`, `mistral`, or `inconclusive → local`, with reasons.

## Acceptance criteria

- Reuse the existing statistics functions; no numpy/scipy.
- Golden-value tests for every metric and the decision rule.
