# T22: Layer 1 OCR engines

| Item | Content |
|---|---|
| Phase | 4 Two-layer redesign |
| ADR | ADR-0002 §1, §6, §8 |
| Depends on | T21 |
| Followed by | T24, T25 |
| Main outputs | `src/ocrbench/ocr_engines.py` (Tesseract, PaddleOCR, Qwen-VL, Mistral), `docs/windows-setup.md` §11 |

## Goal

Implement the Layer 1 candidates A1–A4 behind the `OcrEngine` protocol.

## Implementation

1. No new dependencies (ADR-0002 §8): Tesseract via its command-line program,
   Mistral via `urllib`, PaddleOCR imported lazily. A missing engine gives a clear
   error pointing to `docs/windows-setup.md`.
2. Tesseract (A1): Japanese + English language data; record engine and language versions.
3. PaddleOCR (A2): Japanese model; record model version.
4. Qwen-VL transcribe-only (A3): existing Ollama adapter, a fixed transcription prompt
   that asks for the text only, no interpretation.
5. Mistral OCR (A4): refuses to run unless `OCRBENCH_ALLOW_CLOUD=1`, the API key env var,
   and `--confirm-cloud` are all present. Never log or persist the key. Cost per page
   from configuration.
6. Real-engine tests are marked (`ollama` or a new `ocr_engines` marker) and excluded
   from CI; CI uses fakes only.

## Acceptance criteria

- Tests prove the Mistral lock refuses each missing condition.
- `uv sync` alone still passes all four quality gates; `uv.lock` is unchanged.
