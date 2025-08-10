from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  from doctr.io import DocumentFile  # type: ignore
  from doctr.models import ocr_predictor  # type: ignore
  _HAS_DOCTR = True
except Exception:  # pragma: no cover
  _HAS_DOCTR = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class DocTRAdapter(RecognizerAdapter):
  name = "doctr"
  supports_pdf = True
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_DOCTR:
      raise ImportError(
          "DocTR not available. Install extras: pip install .[ingest-doctr]"
      )

  def _build(self, cfg: dict[str, Any] | None) -> Any:
    reco_arch = (cfg or {}).get("reco_arch", "crnn_vgg16_bn"); det_arch = (cfg or {}).get("det_arch", "db_resnet50")
    pretrained = bool((cfg or {}).get("pretrained", True))
    return ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)

  def recognize_pdf(self, path: str, cfg: dict[str, Any] | None = None) -> RecognizeResult:
    predictor = self._build(cfg)
    t0 = time.time()
    doc = DocumentFile.from_pdf(path)
    res = predictor(doc)
    elapsed = int((time.time() - t0) * 1000)
    lines: list[str] = []
    try:
      for page in res.export()["pages"]:
        for block in page.get("blocks", []):
          for line in block.get("lines", []):
            parts = [w.get("value", "") for w in line.get("words", [])]
            if parts:
              lines.append(" ".join(parts))
    except Exception:
      pass
    text = "\n".join(lines)
    return RecognizeResult(
        text=text,
        mean_confidence=0.0,  # doctr export dict lacks unified confidence
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(doc)},
    )

  def recognize_images(
      self, images: Sequence[Image], cfg: dict[str, Any] | None = None
  ) -> RecognizeResult:
    predictor = self._build(cfg)
    t0 = time.time()
    lines: list[str] = []
    for im in images:
      res = predictor([im])
      try:
        for page in res.export()["pages"]:
          for block in page.get("blocks", []):
            for line in block.get("lines", []):
              parts = [w.get("value", "") for w in line.get("words", [])]
              if parts:
                lines.append(" ".join(parts))
      except Exception:
        continue
    elapsed = int((time.time() - t0) * 1000)
    return RecognizeResult(
        text="\n".join(lines),
        mean_confidence=0.0,
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images)},
    )


def create() -> DocTRAdapter:
  return DocTRAdapter() 