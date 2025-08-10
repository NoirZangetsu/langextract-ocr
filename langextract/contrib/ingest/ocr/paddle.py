from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  from paddleocr import PaddleOCR  # type: ignore
  _HAS_PADDLE = True
except Exception:  # pragma: no cover
  _HAS_PADDLE = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class PaddleAdapter(RecognizerAdapter):
  name = "paddle"
  supports_pdf = True
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_PADDLE:
      raise ImportError(
          "PaddleOCR not available. Install extras: pip install .[ingest-paddle]"
      )

  def _build(self, cfg: dict[str, Any] | None) -> PaddleOCR:
    lang = (cfg or {}).get("lang", "turkish")
    use_gpu = bool((cfg or {}).get("use_gpu", False))
    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:
    ocr = self._build(cfg)
    t0 = time.time()
    result = ocr.ocr(path, cls=True)
    elapsed = int((time.time() - t0) * 1000)
    texts: list[str] = []
    scores: list[float] = []
    for page in result or []:
      for _, (txt, score) in (page or []):
        texts.append(txt)
        try:
          scores.append(float(score))
        except Exception:
          continue
    text = "\n".join(texts)
    mean_conf = sum(scores) / len(scores) if scores else 0.0
    return RecognizeResult(
        text=text,
        mean_confidence=max(0.0, min(1.0, mean_conf)),
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(result or []), "lang": (cfg or {}).get("lang")},
    )

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    ocr = self._build(cfg)
    t0 = time.time()
    texts: list[str] = []
    scores: list[float] = []
    for im in images:
      out = ocr.ocr(im, cls=True)
      for _, (txt, score) in (out or []):
        texts.append(txt)
        try:
          scores.append(float(score))
        except Exception:
          continue
    elapsed = int((time.time() - t0) * 1000)
    mean_conf = sum(scores) / len(scores) if scores else 0.0
    return RecognizeResult(
        text="\n".join(texts),
        mean_confidence=max(0.0, min(1.0, mean_conf)),
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images), "lang": (cfg or {}).get("lang")},
    )


def create() -> PaddleAdapter:
  return PaddleAdapter() 