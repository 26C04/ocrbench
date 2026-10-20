# T20: EDI field schema and normalization

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §2 |
| Depends on | T19 |
| Followed by | T21, T23 |
| Main outputs | `src/ocrbench/edi_fields.py`, `src/ocrbench/customers.py` (or a data file), tests |

## Goal

Define the 4-field record every pipeline returns, and the normalization used
before comparison.

## Implementation

1. `EdiFields` (pydantic, `extra="forbid"`, `strict=True`):
   `customer: str | None`, `part_no: str | None`, `order_no: str | None`,
   `quantity: int | None` (≥ 1). Provide `edi_format_schema()` for Ollama `format`.
2. Customer mapping: a versioned, committed table from printed customer names to
   EDI site IDs (first entry: `Nidecマシンツール株式会社` → `hwao_nmtj`). Use only
   synthetic or already-public names. Unmapped names return `None`.
3. `normalize_order_no(value)`: strip outer whitespace; the `-YYYY` suffix rule is
   controlled by one constant reflecting ADR-0002 D1. Until D1 is settled, default
   to keeping the value as printed and make the rule switchable in config.
4. Quantity: extend the existing ground-truth helper so `"1 PC"`, `"1PC"`, `"4PC"`,
   `"12 pcs"`, `"３個"` normalize correctly. Fix the current bug where `"1 PC"` → `None`.
5. `comparable_fields(fields)` returns the normalized tuple used by scoring.
6. Do not modify or remove `OrderDocument`; pipeline B maps it to `EdiFields`.

## Acceptance criteria

- Unknown keys, text around JSON, and non-integer quantities are rejected.
- All normalization rules in ADR-0002 §2 have tests, including both D1 settings.

## Tests

- Table-driven tests for quantity units with and without spaces, full-width digits,
  separators, zero and negative values.
- Customer mapping hit, miss, and whitespace variants.
