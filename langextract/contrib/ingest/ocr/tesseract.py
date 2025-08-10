from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  import pytesseract  # type: ignore
  _HAS_TESS = True
except Exception:  # pragma: no cover
  _HAS_TESS = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class TesseractAdapter(RecognizerAdapter):
  name = "tesseract"
  supports_pdf = False  # via images only
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_TESS:
      raise ImportError(
          "Tesseract not available. Install tesseract system package and pip install .[ingest-tesseract]"
      )

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:  # pragma: no cover - not used directly
    raise NotImplementedError("Use recognize_images() after rasterizing PDF")

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    lang = (cfg or {}).get("lang", "tur")
    t0 = time.time()
    texts: list[str] = []
    confs: list[float] = []
    for im in images:
      txt = pytesseract.image_to_string(im, lang=lang)
      texts.append(txt)
      try:
        data = pytesseract.image_to_data(im, lang=lang, output_type=pytesseract.Output.DICT)
        conf_vals = [float(c) for c in data.get("conf", []) if c not in ("-1", None)]
        if conf_vals:
          confs.append(sum(conf_vals) / (100.0 * len(conf_vals)))
      except Exception:
        continue
    elapsed = int((time.time() - t0) * 1000)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return RecognizeResult(
        text="\n".join(texts),
        mean_confidence=max(0.0, min(1.0, mean_conf)),
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images), "lang": lang},
    )


def create() -> TesseractAdapter:
  return TesseractAdapter() 