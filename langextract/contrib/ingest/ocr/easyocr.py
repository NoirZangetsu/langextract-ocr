from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  import easyocr  # type: ignore
  _HAS_EASY = True
except Exception:  # pragma: no cover
  _HAS_EASY = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class EasyOCRAdapter(RecognizerAdapter):
  name = "easyocr"
  supports_pdf = False
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_EASY:
      raise ImportError(
          "EasyOCR not available. Install extras: pip install .[ingest-easyocr]"
      )

  def _build(self, cfg: dict[str, Any] | None) -> Any:
    lang = (cfg or {}).get("lang", "tr")
    gpu = bool((cfg or {}).get("use_gpu", False))
    return easyocr.Reader([lang], gpu=gpu)

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:  # pragma: no cover - not used directly
    raise NotImplementedError("Use recognize_images() after rasterizing PDF")

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    reader = self._build(cfg)
    t0 = time.time()
    texts: list[str] = []
    confs: list[float] = []
    for im in images:
      res = reader.readtext(im)
      for _, txt, score in res:
        texts.append(txt)
        try:
          confs.append(float(score))
        except Exception:
          continue
    elapsed = int((time.time() - t0) * 1000)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return RecognizeResult(
        text="\n".join(texts),
        mean_confidence=max(0.0, min(1.0, mean_conf)),
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images), "lang": (cfg or {}).get("lang")},
    )


def create() -> EasyOCRAdapter:
  return EasyOCRAdapter() 