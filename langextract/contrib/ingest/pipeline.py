from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .. import ingest  # type: ignore  # for namespace resolution
from .types import RecognizeResult, RecognizerAdapter
from . import autodeps

# Optional deps for PDF vector text and rasterization
try:
  from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
  _HAS_PDFMINER = True
except Exception:  # pragma: no cover
  _HAS_PDFMINER = False
  pdfminer_extract_text = None  # type: ignore

try:
  from pdf2image import convert_from_path  # type: ignore
  _HAS_PDF2IMAGE = True
except Exception:  # pragma: no cover
  _HAS_PDF2IMAGE = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


@dataclass
class IngestResult:
  input_path: str | None
  used_ocr: bool
  engine: str | None
  ocr_confidence: float | None
  text: str
  meta: dict[str, Any]


def is_sufficient_vector_text(text: str, min_len: int = 64) -> bool:
  """Heuristic to decide whether vector-extracted text is sufficient."""
  if not text:
    return False
  # Require minimal length and presence of non-ASCII for Turkish when likely
  return len(text.strip()) >= min_len


def _load_images_from_pdf(
    path: str,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Image]:
  if not _HAS_PDF2IMAGE:
    raise ImportError(
        "pdf2image not available. Install extras: pip install .[ingest]"
    )
  return convert_from_path(
      path, dpi=dpi, first_page=first_page, last_page=last_page
  )


def _pilot_crop(images: Sequence[Image], max_side: int = 512) -> list[Image]:
  out: list[Image] = []
  for im in images[:1]:
    w, h = im.size
    box = (0, 0, min(w, max_side), min(h, max_side))
    out.append(im.crop(box))
  return out or list(images[:1])


def build_adapter(engine: str) -> RecognizerAdapter:
  e = engine.lower()
  if e == "paddle":
    from .ocr.paddle import create
    return create()
  if e == "tesseract":
    from .ocr.tesseract import create
    return create()
  if e == "easyocr":
    from .ocr.easyocr import create
    return create()
  if e == "doctr":
    from .ocr.doctr import create
    return create()
  if e == "trocr":
    from .vlm.trocr import create
    return create()
  if e == "donut":
    from .vlm.donut import create
    return create()
  if e == "nougat":
    from .vlm.nougat import create
    return create()
  raise ValueError(f"Unknown engine: {engine}")


def auto_select_engine(
    images: Sequence[Image],
    candidates: Sequence[str] = ("paddle", "easyocr", "tesseract"),
    cfg: dict[str, Any] | None = None,
) -> str:
  """Run a fast pilot on first crop page to pick an engine by score & charset."""
  pilot_images = _pilot_crop(images)
  best_name = None
  best_score = -1.0
  for name in candidates:
    try:
      adapter = build_adapter(name)
      res = adapter.recognize_images(pilot_images, cfg)
      score = res.mean_confidence
      txt = (res.text or "")[:512]
      # Reward presence of Turkish-specific characters when lang suggests Turkish
      if any(ch in txt for ch in "çğıöşüÇĞİÖŞÜ"):
        score += 0.05
      if score > best_score:
        best_score = score
        best_name = name
    except Exception:
      continue
  return best_name or candidates[0]


def recognize_pdf(
    path: str,
    engine: str = "auto",
    prefer_ocr: bool = False,
    ocr_cfg: dict[str, Any] | None = None,
    dpi: int = 300,
    first_page: int | None = None,
    last_page: int | None = None,
    confidence_threshold: float = 0.6,
    fallback_order: Sequence[str] | None = None,
) -> IngestResult:
  ocr_cfg = ocr_cfg or {}
  # Ensure common deps
  autodeps.ensure_common()
  meta: dict[str, Any] = {"pages": None}

  # Step 1: Try vector text if allowed
  vector_text = ""
  if not prefer_ocr and _HAS_PDFMINER:
    try:
      vector_text = pdfminer_extract_text(path)
    except Exception:
      vector_text = ""
  if vector_text and is_sufficient_vector_text(vector_text):
    return IngestResult(
        input_path=path,
        used_ocr=False,
        engine=None,
        ocr_confidence=None,
        text=vector_text,
        meta={"vector": True},
    )

  # Step 2: Rasterize
  images = _load_images_from_pdf(path, dpi=dpi, first_page=first_page, last_page=last_page)

  # Step 3: Choose engine
  if engine == "auto":
    engine = auto_select_engine(images, cfg=ocr_cfg)

  # Step 4: Run with fallback
  order = list(fallback_order) if fallback_order else [engine]
  if engine not in order:
    order.insert(0, engine)
  # default fallback if not provided
  for name in ("paddle", "easyocr", "tesseract"):
    if name not in order:
      order.append(name)

  last_res: RecognizeResult | None = None
  for name in order:
    try:
      # ensure engine-specific deps
      autodeps.ensure_for_engine(name)
      adapter = build_adapter(name)
      if adapter.supports_pdf:
        res = adapter.recognize_pdf(path, ocr_cfg)
      else:
        res = adapter.recognize_images(images, ocr_cfg)
      last_res = res
      if res.mean_confidence >= confidence_threshold:
        return IngestResult(
            input_path=path,
            used_ocr=True,
            engine=name,
            ocr_confidence=res.mean_confidence,
            text=res.text,
            meta=res.meta,
        )
    except Exception:
      continue

  # Fallback to last result if any
  if last_res is not None:
    return IngestResult(
        input_path=path,
        used_ocr=True,
        engine=last_res.engine,
        ocr_confidence=last_res.mean_confidence,
        text=last_res.text,
        meta=last_res.meta,
    )

  raise RuntimeError("No OCR engine succeeded for given PDF")


def recognize_images(
    images: Sequence[Image],
    engine: str = "auto",
    ocr_cfg: dict[str, Any] | None = None,
    confidence_threshold: float = 0.6,
) -> IngestResult:
  ocr_cfg = ocr_cfg or {}
  autodeps.ensure_common()
  if engine == "auto":
    engine = auto_select_engine(images, cfg=ocr_cfg)
  order = [engine]
  for name in ("paddle", "easyocr", "tesseract"):
    if name not in order:
      order.append(name)

  last_res: RecognizeResult | None = None
  for name in order:
    try:
      autodeps.ensure_for_engine(name)
      adapter = build_adapter(name)
      res = adapter.recognize_images(images, ocr_cfg)
      last_res = res
      if res.mean_confidence >= confidence_threshold:
        return IngestResult(
            input_path=None,
            used_ocr=True,
            engine=name,
            ocr_confidence=res.mean_confidence,
            text=res.text,
            meta=res.meta,
        )
    except Exception:
      continue

  if last_res is not None:
    return IngestResult(
        input_path=None,
        used_ocr=True,
        engine=last_res.engine,
        ocr_confidence=last_res.mean_confidence,
        text=last_res.text,
        meta=last_res.meta,
    )
  raise RuntimeError("No OCR engine succeeded for images")


def maybe_enable_htr(
    images: Sequence[Image],
    enable_htr: bool = False,
    printed_text_engine: str = "paddle",
    htr_engine: str = "trocr",
    ocr_cfg: dict[str, Any] | None = None,
) -> IngestResult:
  """Very simple heuristic: if enable_htr, run printed OCR; else only printed.

  Real layout & handwriting segmentation is out of scope for basic contrib.
  """
  if not enable_htr:
    return recognize_images(images, engine=printed_text_engine, ocr_cfg=ocr_cfg)

  # Try printed then HTR and concatenate, keeping line/page breaks
  printed = recognize_images(images, engine=printed_text_engine, ocr_cfg=ocr_cfg)
  try:
    htr_adapter = build_adapter(htr_engine)
    htr_res = htr_adapter.recognize_images(images, ocr_cfg)
    combined_text = printed.text + "\n" + htr_res.text
    mean_conf = max(printed.ocr_confidence or 0.0, htr_res.mean_confidence)
    return IngestResult(
        input_path=None,
        used_ocr=True,
        engine=f"{printed.engine}+{htr_res.engine}",
        ocr_confidence=mean_conf,
        text=combined_text,
        meta={"printed": printed.meta, "htr": htr_res.meta},
    )
  except Exception:
    return printed 