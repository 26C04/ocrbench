# T24: Runner, CLI, and reports for pipelines

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §1, §3, §4 |
| Depends on | T21, T22, T23 |
| Followed by | T25, H6 |
| Main outputs | runner/results/reporting/cli changes, dataset split checks |

## Goal

Run any pipeline through the existing runner and report ADR-0002 metrics.

## Implementation

1. Runner accepts a pipeline configuration (ID, engine, model, prompt, command) and
   records it, with versions/digests, in the run manifest and `results.json`.
2. CLI: `ocrbench run --pipeline <config>`; `--confirm-cloud` for A4. Existing
   commands keep working unchanged.
3. Dataset: replace the fixed 40/20/40 strict check with a configurable minimum for
   `dev` and `final`; keep the existing strict mode available.
4. `report.md` and the anonymous summary add the ADR-0002 metrics. Detail reports
   stay local, as before.
5. `compare-runs` reports the paired comparison and the decision rule.

## Acceptance criteria

- Fake end-to-end test: two fake pipelines on a synthetic dataset produce a report
  with all ADR-0002 metrics.
- Existing CLI tests still pass.
