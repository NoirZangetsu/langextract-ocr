from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

try:
  from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dep
  Image = Any  # type: ignore


@dataclass
class RecognizeResult:
  """Unified recognition result returned by all adapters.

  Attributes:
    text: The concatenated plain text extracted from all pages/images.
    mean_confidence: Mean confidence in [0,1]. If engine does not provide,
      populate with a heuristic estimation or 0.0.
    engine: Engine name (e.g., "paddle", "tesseract", "easyocr", "doctr",
      "trocr", "donut", "nougat").
    meta: Arbitrary metadata such as page_count, elapsed_ms, device, lang, etc.
  """

  text: str
  mean_confidence: float
  engine: str
  meta: dict[str, Any]


class RecognizerAdapter(Protocol):
  """Adapter interface for OCR/VLM/HTR engines."""

  name: str
  supports_pdf: bool
  supports_image: bool

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:
    ...

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    ...


@dataclass
class EngineSpec:
  name: str
  adapter: RecognizerAdapter
  priority: int = 0  # smaller means tried earlier in default order 