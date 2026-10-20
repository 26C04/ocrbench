# T19: ADR-0002 and project documentation

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 (all) |
| Depends on | T18 |
| Followed by | T20–T25 |
| Main outputs | `adr/0002-two-layer-edi-field-benchmark.md`, `CLAUDE.md`, `tasks/T19`–`T25`, `tasks/README.md` |

## Goal

Record the two-layer, 4-field design before any code changes, without removing
anything that exists.

## Implementation

1. Tag the pre-ADR-0002 state as `v0.1-vlm-benchmark` (annotated tag on the T18 merge).
2. Add ADR-0002 with Status `Proposed` and the open decisions D1–D5.
3. Add a "Revised by" line to ADR-0001's header. Do not change its body.
4. Add `CLAUDE.md` with the project context and a note that ADRs take precedence.
5. Add task files T20–T25 and human tasks H6–H8 to the task list.

## Acceptance criteria

- No source or test file changes. All four quality gates still pass.
- ADR-0001's body is unchanged.
