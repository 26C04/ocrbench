# T21: Pipeline interface and adapters

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §1 |
| Depends on | T20 |
| Followed by | T22, T24 |
| Main outputs | `src/ocrbench/pipelines/` (protocol, OCR+LLM, vision, external command, fake) |

## Goal

Let `ocrbench` run any candidate pipeline through one interface.

## Implementation

1. `Pipeline` protocol: `extract(image: PreprocessResult) -> PipelineResult`.
   `PipelineResult` holds: `fields: EdiFields | None`, `status` (`ok`, `needs_review`,
   `format_error`, `error`), `layer1_text: str | None`, timings (layer 1, layer 2, total),
   `cost_usd: float`, and an identifier with versions/digests for the manifest.
2. `OcrLlmPipeline(ocr: OcrEngine, llm: OllamaAdapter, prompt)`: Layer 1 then Layer 2.
   Layer 2 uses the ADR-0001 §7 protocol and `edi_format_schema()`; one retry on
   schema failure without hints.
3. `VisionPipeline`: wraps the existing vision-model path (candidate B) and maps
   `OrderDocument` to `EdiFields` (single item only; several items → `needs_review`).
4. `ExternalCommandPipeline` (candidate C): runs a configured command with the image
   path, reads one JSON object from stdout:
   `{"status": "ok"|"needs_review", "fields": {...} | null, "layer1_text": str | null}`.
   Timeout, non-zero exit, and invalid JSON become `error`. Never pass secrets in argv.
5. `OcrEngine` protocol: `transcribe(image) -> OcrResult(text, seconds, cost_usd, engine_id)`.
6. Fakes for every protocol for CI.

## Acceptance criteria

- No new required runtime dependency.
- All adapters are covered by fake-based tests; no network in normal tests.
