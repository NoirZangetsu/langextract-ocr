import json
from pathlib import Path
from types import SimpleNamespace

import langextract as lx
from langextract.contrib.ingest import cli as ingest_cli
from langextract.contrib.ingest.pipeline import IngestResult


def _make_prompt(tmp_path: Path) -> Path:
  p = tmp_path / "prompt.txt"
  p.write_text("Extract entities", encoding="utf-8")
  return p


def _make_examples(tmp_path: Path) -> Path:
  p = tmp_path / "examples.json"
  payload = [
    {
      "text": "John is 30.",
      "extractions": [
        {"extraction_class": "person", "extraction_text": "John"}
      ],
    }
  ]
  p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
  return p


def test_cli_pdf_outputs_expected_json(monkeypatch, tmp_path):
  # Stub pipeline.recognize_pdf
  def fake_recognize_pdf(path, **kwargs):
    return IngestResult(
      input_path=path,
      used_ocr=True,
      engine="paddle",
      ocr_confidence=0.87,
      text="hello world",
      meta={}
    )
  monkeypatch.setattr(ingest_cli, "recognize_pdf", fake_recognize_pdf)

  # Stub lx.extract to avoid LLM
  def fake_extract(**kwargs):
    return lx.data.AnnotatedDocument(text="hello world")
  monkeypatch.setattr(lx, "extract", fake_extract)

  prompt = _make_prompt(tmp_path)
  examples = _make_examples(tmp_path)
  out = tmp_path / "out.json"

  rc = ingest_cli.main([
    "pdf", "dummy.pdf",
    "--engine", "auto",
    "--prompt", str(prompt),
    "--few-shots-json", str(examples),
    "--output", str(out),
  ])
  assert rc == 0
  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["used_ocr"] is True
  assert data["engine"] == "paddle"
  assert data["ocr_confidence"] == 0.87
  assert data["text_len"] == len("hello world")
  assert "extraction" in data


def test_cli_image_outputs_expected_json(monkeypatch, tmp_path):
  # Create a fake image file
  img_path = tmp_path / "img.png"
  img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal header; PIL open will fail; we stub recognize_images anyway

  # Stub pipeline.recognize_images
  def fake_recognize_images(images, **kwargs):
    return IngestResult(
      input_path=None,
      used_ocr=True,
      engine="easyocr",
      ocr_confidence=0.75,
      text="image text",
      meta={}
    )
  monkeypatch.setattr(ingest_cli, "recognize_images", fake_recognize_images)

  # Stub PIL.Image.open used in CLI to avoid decoding
  class FakeImage:
    pass
  import langextract.contrib.ingest.cli as _cli
  monkeypatch.setattr(_cli, "Image", SimpleNamespace(open=lambda p: FakeImage()))

  # Stub lx.extract
  def fake_extract(**kwargs):
    return lx.data.AnnotatedDocument(text="image text")
  monkeypatch.setattr(lx, "extract", fake_extract)

  prompt = _make_prompt(tmp_path)
  examples = _make_examples(tmp_path)

  rc = ingest_cli.main([
    "image", str(img_path),
    "--engine", "easyocr",
    "--prompt", str(prompt),
    "--few-shots-json", str(examples),
  ])
  assert rc == 0 