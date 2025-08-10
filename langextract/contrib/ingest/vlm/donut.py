from __future__ import annotations

import time
from typing import Any, Sequence

from ..types import RecognizerAdapter, RecognizeResult

try:
  import torch  # type: ignore
  from transformers import DonutProcessor, VisionEncoderDecoderModel  # type: ignore
  _HAS_DONUT = True
except Exception:  # pragma: no cover
  _HAS_DONUT = False

try:
  from PIL import Image
except Exception:  # pragma: no cover
  Image = Any  # type: ignore


class DonutAdapter(RecognizerAdapter):
  name = "donut"
  supports_pdf = False  # requires image conversion first
  supports_image = True

  def __init__(self) -> None:
    if not _HAS_DONUT:
      raise ImportError(
          "Donut not available. Install extras: pip install .[ingest-vlm]"
      )

  def _build(self, cfg: dict[str, Any] | None) -> tuple[Any, Any, str]:
    model_id = (cfg or {}).get("model", "naver-clova-ix/donut-base")
    device = "cuda" if torch.cuda.is_available() and (cfg or {}).get("use_gpu", True) else "cpu"
    processor = DonutProcessor.from_pretrained(model_id)
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
      task_prompt = (cfg or {}).get("task_prompt", "<s_docvqa><s_question>extract_text</s_question><s_answer>")
      decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
      outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=1024)
      seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
      texts.append(seq)
    elapsed = int((time.time() - t0) * 1000)
    return RecognizeResult(
        text="\n".join(texts),
        mean_confidence=0.0,
        engine=self.name,
        meta={"elapsed_ms": elapsed, "pages": len(images), "device": device},
    )


def create() -> DonutAdapter:
  return DonutAdapter() 