"""Phase 2: verify that the part in a photo is the part its sheet describes.

There is no CAD database yet, so the sheet itself is the reference. It prints
the part name, the material, the raw stock size ``φD × L`` the part was cut
from, and features such as through holes. A finished part must fit inside its
stock and be close to it, so a photo with a known scale can be checked against
the sheet with plain geometry. Every part a worker confirms is stored in a
registry, so repeat part numbers are also compared with earlier shipments.

Modules:

- :mod:`ocrbench.parts.spec` reads the part spec from the sheet text.
- :mod:`ocrbench.parts.vision` finds the scale marker, segments the part and
  measures it in millimetres.
- :mod:`ocrbench.parts.sam3` is the SAM3 segmenter (needs ``transformers`` and
  ``torch``, installed on the GPU PC only).
- :mod:`ocrbench.parts.check` applies the rules and gives pass, fail or
  cannot-check, always with reasons.
- :mod:`ocrbench.parts.registry` stores confirmed parts (no images).
- :mod:`ocrbench.parts.calibrate` sets the tolerances from confirmed parts.
- :mod:`ocrbench.parts.barcode` decodes the 伝票No and other barcodes.
- :mod:`ocrbench.parts.synthetic` draws invented part photos for tests and demos.

See ``docs/part-verification.md``.
"""
