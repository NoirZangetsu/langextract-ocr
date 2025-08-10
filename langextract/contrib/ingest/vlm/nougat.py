from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  # nougat provides CLI and python interfaces; prefer python if installed
  from nougat import NougatModel  # type: ignore
  _HAS_NOUGAT = True
except Exception:  # pragma: no cover
  _HAS_NOUGAT = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class NougatAdapter(RecognizerAdapter):
  name = "nougat"
  supports_pdf = True
  supports_image = False

  def __init__(self) -> None:
    if not _HAS_NOUGAT:
      raise ImportError(
          "Nougat not available. Install extras: pip install .[ingest-vlm]"
      )

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:
    t0 = time.time()
    model_name = (cfg or {}).get("model", "facebook/nougat-base")
    device = (cfg or {}).get("device", "cuda")
    model = NougatModel.from_pretrained(model_name).to(device)
    text = model.inference(path)  # returns Markdown-like text
    elapsed = int((time.time() - t0) * 1000)
    return RecognizeResult(
        text=text,
        mean_confidence=0.0,
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": None, "device": device},
    )

  def recognize_images(self, images: Sequence[Image], cfg: dict[str, Any] | None = None) -> RecognizeResult:  # pragma: no cover
    raise NotImplementedError("Nougat expects PDF inputs; convert images to PDF first.")


def create() -> NougatAdapter:
  return NougatAdapter() 