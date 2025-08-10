from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import langextract as lx
from .pipeline import recognize_pdf, recognize_images, IngestResult

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


def _add_common_args(p: argparse.ArgumentParser) -> None:
  p.add_argument("--engine", default="auto", help="paddle|tesseract|easyocr|doctr|trocr|donut|nougat|auto")
  p.add_argument("--enable-htr", action="store_true", help="Enable HTR path (TrOCR)")
  p.add_argument("--ocr-lang", default=None, help="Language hint for OCR engines (e.g., turkish|tr|multilingual)")
  p.add_argument("--prefer-ocr", action="store_true", help="Skip vector text and force OCR")
  p.add_argument("--dpi", type=int, default=300)
  p.add_argument("--first-page", type=int, default=None)
  p.add_argument("--last-page", type=int, default=None)
  p.add_argument("--max-vector-pages", type=int, default=None)
  p.add_argument("--confidence-threshold", type=float, default=0.6)
  p.add_argument("--prompt", type=str, default=None)
  p.add_argument("--schema-json", type=str, default=None)
  p.add_argument("--few-shots-json", type=str, default=None)
  p.add_argument("--model-id", type=str, default="gemini-2.5-flash")
  p.add_argument("--api-key", type=str, default=None)
  p.add_argument("--output", type=str, default=None, help="Path to write JSON output")
  p.epilog = (
    "Note: Missing OCR/VLM dependencies will be auto-installed by default. "
    "To disable, set LANGEXTRACT_INGEST_AUTO_INSTALL=0."
  )


def _build_examples(path: str | None) -> list[lx.data.ExampleData]:
  if not path:
    return []
  content = Path(path).read_text(encoding="utf-8")
  data = json.loads(content)
  examples: list[lx.data.ExampleData] = []
  for item in data:
    exts = []
    for ext in item.get("extractions", []):
      exts.append(
          lx.data.Extraction(
              extraction_class=ext["extraction_class"],
              extraction_text=ext["extraction_text"],
              attributes=ext.get("attributes"),
          )
      )
    examples.append(lx.data.ExampleData(text=item["text"], extractions=exts))
  return examples


def _ocr_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
  cfg: dict[str, Any] = {}
  if args.ocr_lang:
    # Normalize a few typical aliases
    lang = args.ocr_lang.lower()
    cfg["lang"] = {"tr": "turkish", "turkish": "turkish"}.get(lang, lang)
  cfg["use_gpu"] = True  # let engines downscale to CPU if unavailable
  return cfg


def _emit_output(payload: dict[str, Any], output_path: str | None) -> None:
  text = json.dumps(payload, ensure_ascii=False, indent=2)
  if output_path:
    Path(output_path).write_text(text, encoding="utf-8")
  else:
    print(text)


def cmd_pdf(args: argparse.Namespace) -> int:
  ocr_cfg = _ocr_cfg_from_args(args)
  ing = recognize_pdf(
      path=args.path,
      engine=args.engine,
      prefer_ocr=args.prefer_ocr,
      ocr_cfg=ocr_cfg,
      dpi=args.dpi,
      first_page=args.first_page,
      last_page=args.last_page,
      confidence_threshold=args.confidence_threshold,
  )

  prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None
  examples = _build_examples(args.few_shots_json)
  if not examples:
    raise SystemExit("Examples are required. Provide --few-shots-json")
  schema_json = Path(args.schema_json).read_text(encoding="utf-8") if args.schema_json else None

  extraction = lx.extract(
      text_or_documents=ing.text,
      prompt_description=prompt,
      examples=examples,
      model_id=args.model_id,
      api_key=args.api_key,
  )

  payload = {
      "input_path": args.path,
      "used_ocr": ing.used_ocr,
      "engine": ing.engine,
      "ocr_confidence": ing.ocr_confidence,
      "text_len": len(ing.text or ""),
      "extraction": lx.data_lib.annotated_document_to_dict(extraction),
  }
  _emit_output(payload, args.output)
  return 0


def cmd_image(args: argparse.Namespace) -> int:
  image_paths = [args.path] if Path(args.path).is_file() else sorted(str(p) for p in Path(args.path).glob("*.png"))
  images = [Image.open(p) for p in image_paths]
  ocr_cfg = _ocr_cfg_from_args(args)
  ing = recognize_images(images, engine=args.engine, ocr_cfg=ocr_cfg, confidence_threshold=args.confidence_threshold)

  prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else None
  examples = _build_examples(args.few_shots_json)
  if not examples:
    raise SystemExit("Examples are required. Provide --few-shots-json")

  extraction = lx.extract(
      text_or_documents=ing.text,
      prompt_description=prompt,
      examples=examples,
      model_id=args.model_id,
      api_key=args.api_key,
  )

  payload = {
      "input_path": args.path,
      "used_ocr": ing.used_ocr,
      "engine": ing.engine,
      "ocr_confidence": ing.ocr_confidence,
      "text_len": len(ing.text or ""),
      "extraction": lx.data_lib.annotated_document_to_dict(extraction),
  }
  _emit_output(payload, args.output)
  return 0


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(prog="langextract-ingest", description="Ingest PDF/Images with OCR/VLM and run LangExtract")
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_pdf = sub.add_parser("pdf", help="Ingest a PDF file")
  p_pdf.add_argument("path", help="Path to PDF")
  _add_common_args(p_pdf)
  p_pdf.set_defaults(func=cmd_pdf)

  p_img = sub.add_parser("image", help="Ingest a single image or a directory of images (.png)")
  p_img.add_argument("path", help="Path to image file or directory of images")
  _add_common_args(p_img)
  p_img.set_defaults(func=cmd_image)

  args = parser.parse_args(argv)
  try:
    return args.func(args)
  except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.stderr.write("Hint: install required extras, e.g. pip install .[ingest,ingest-paddle]\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main()) 