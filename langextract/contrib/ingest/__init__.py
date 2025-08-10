"""OCR/VLM ingestion layer for PDF and image inputs.

This package provides optional pre-processing to turn PDFs and images into
text for LangExtract, via pluggable recognizer adapters (OCR/VLM/HTR).
"""

from __future__ import annotations

from . import pipeline  # noqa: F401
from . import types  # noqa: F401

__all__ = [
    "pipeline",
    "types",
] 