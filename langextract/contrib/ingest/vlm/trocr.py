from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  import torch  # type: ignore
  from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
  _HAS_TROCR = True
except Exception:  # pragma: no cover
  _HAS_TROCR = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class TrOCRAdapter(RecognizerAdapter):
  name = "trocr"
  supports_pdf = False
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_TROCR:
      raise ImportError(
          "TrOCR not available. Install extras: pip install .[ingest-vlm]"
      )

  def _build(self, cfg: dict[str, Any] | None) -> tuple[Any, Any, str]:
    model_id = (cfg or {}).get("model", "microsoft/trocr-base-printed")
    device = "cuda" if torch.cuda.is_available() and (cfg or {}).get("use_gpu", True) else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    return processor, model, device

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:  # pragma: no cover - must rasterize
    raise NotImplementedError("Use recognize_images() after rasterizing PDF")

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    processor, model, device = self._build(cfg)
    t0 = time.time()
    texts: list[str] = []
    for im in images:
      pixel_values = processor(images=im, return_tensors="pt").pixel_values.to(device)
      generated_ids = model.generate(pixel_values)
      txt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
      texts.append(txt)
    elapsed = int((time.time() - t0) * 1000)
    return RecognizeResult(
        text="\n".join(texts),
        mean_confidence=0.0,  # not provided directly
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images), "device": device},
    )


def create() -> TrOCRAdapter:
  return TrOCRAdapter() 