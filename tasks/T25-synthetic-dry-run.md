# T25: Synthetic dry run

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §1, §3 |
| Depends on | T24 |
| Followed by | H6 |
| Main outputs | synthetic sheet generator (test code), `docs/` run instructions |

## Goal

Prove every local candidate runs end to end before real sheets arrive. Results from
synthetic data are not benchmark results.

## Implementation

1. Generate synthetic sheets in code (layout similar to the real sheet, invented values),
   or reuse the teammate's generator if its output is synthetic only.
2. Run A1–A3, B, and C (if its command contract is ready) on the synthetic set on the
   Windows machine; record versions and timings.
3. Document the commands in `docs/`.

## Acceptance criteria

- One report per local candidate from a synthetic run, kept outside the repo.
- No real sheet data used or committed.
