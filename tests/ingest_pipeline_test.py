import types
import builtins
import pytest

from langextract.contrib.ingest import pipeline
from langextract.contrib.ingest.types import RecognizerAdapter, RecognizeResult


class DummyAdapter(RecognizerAdapter):
  name = "dummy"
  supports_pdf = False
  supports_image = True

  def recognize_pdf(self, path, cfg=None):  # pragma: no cover
    raise NotImplementedError

  def recognize_images(self, images, cfg=None):
    return RecognizeResult(text="hello", mean_confidence=0.9, engine=self.name, meta={})


def test_auto_select_engine_prefers_higher_confidence(monkeypatch):
  # Build two adapters with different confidences
  class A(RecognizerAdapter):
    name = "a"; supports_pdf = False; supports_image = True
    def recognize_pdf(self, path, cfg=None):
      raise NotImplementedError
    def recognize_images(self, images, cfg=None):
      return RecognizeResult(text="a", mean_confidence=0.6, engine=self.name, meta={})

  class B(RecognizerAdapter):
    name = "b"; supports_pdf = False; supports_image = True
    def recognize_pdf(self, path, cfg=None):
      raise NotImplementedError
    def recognize_images(self, images, cfg=None):
      return RecognizeResult(text="b çğöşü", mean_confidence=0.65, engine=self.name, meta={})

  def build_adapter(name: str):
    return B() if name == "b" else A()

  monkeypatch.setattr(pipeline, "build_adapter", build_adapter)
  # Fake image list
  images = [types.SimpleNamespace(size=(100, 100), crop=lambda box: None)]
  chosen = pipeline.auto_select_engine(images, candidates=("a", "b"))
  assert chosen == "b"


def test_vector_text_bypass(monkeypatch, tmp_path):
  # Create a dummy PDF path (we won't parse it); just force vector text path
  pdf_path = tmp_path / "dummy.pdf"
  pdf_path.write_text("ignored")

  monkeypatch.setattr(pipeline, "_HAS_PDFMINER", True)
  monkeypatch.setattr(pipeline, "pdfminer_extract_text", lambda p: "some vector text" * 10)

  res = pipeline.recognize_pdf(str(pdf_path), engine="auto")
  assert res.used_ocr is False
  assert res.text 