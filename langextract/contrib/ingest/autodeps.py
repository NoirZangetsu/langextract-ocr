from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence

_ENGINE_TO_PACKAGES: dict[str, list[str]] = {
    # core OCR engines
    "paddle": ["paddleocr>=2.7.0.3"],
    "easyocr": ["easyocr>=1.7.1"],
    "tesseract": ["pytesseract>=0.3.10"],  # system tesseract binary still required
    "doctr": ["python-doctr[torch]>=0.8.1"],
    # vlm/htr engines
    "trocr": ["transformers>=4.41.0", "timm>=1.0.0"],
    "donut": ["transformers>=4.41.0", "timm>=1.0.0"],
    "nougat": ["transformers>=4.41.0"],  # nougat python package may differ; prefer manual install
}

_COMMON: list[str] = [
    "pdfminer.six>=20221105",
    "pdf2image>=1.16.3",
    "pillow>=10.0.0",
    "numpy>=1.20.0",
    "tenacity>=8.0.0",
]


def _should_auto_install() -> bool:
    env = os.getenv("LANGEXTRACT_INGEST_AUTO_INSTALL", "1").strip()
    return env not in ("0", "false", "False", "no")


def _pip_install(pkgs: Sequence[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *pkgs]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False


def ensure_common() -> None:
    if not _should_auto_install():
        return
    _pip_install(_COMMON)


def ensure_for_engine(engine: str) -> None:
    if not _should_auto_install():
        return
    pkgs = _ENGINE_TO_PACKAGES.get(engine, [])
    if pkgs:
        _pip_install(pkgs) 