"""PDF validation and page-by-page text extraction with an optional Tesseract OCR fallback."""
from __future__ import annotations

import hashlib
import io
import logging
import shutil
from dataclasses import dataclass, field
from functools import lru_cache

import fitz  # PyMuPDF

from ..config import Settings

log = logging.getLogger("skillpulse.pdf")

PDF_MAGIC = b"%PDF-"


class PDFValidationError(ValueError):
    """Raised when an uploaded file is not a usable PDF."""


@dataclass
class PageResult:
    page_number: int
    text: str
    char_count: int
    method: str      # text | ocr | none
    status: str      # ok | empty | ocr_failed


@dataclass
class ExtractionResult:
    page_count: int
    pages: list[PageResult] = field(default_factory=list)
    ocr_status: str = "not_needed"   # not_needed | used | unavailable | disabled | failed
    ocr_message: str | None = None
    unreadable_without_images: int = 0

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def empty_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.status != "ok"]

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.method == "ocr" and p.status == "ok"]

    @property
    def method(self) -> str:
        methods = {p.method for p in self.pages if p.status == "ok"}
        if methods == {"ocr"}:
            return "ocr"
        if "ocr" in methods:
            return "mixed"
        return "text"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_bytes(data: bytes) -> int:
    """Validate signature and structure. Returns page count or raises PDFValidationError."""
    if not data:
        raise PDFValidationError("The file is empty.")
    head = data[:1024]
    if PDF_MAGIC not in head:
        raise PDFValidationError("The file does not have a PDF signature (%PDF-). Only PDF files are accepted.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises various errors for corrupt input
        raise PDFValidationError(f"The PDF is corrupted or cannot be opened ({exc.__class__.__name__}).") from exc
    try:
        if doc.needs_pass:
            raise PDFValidationError("The PDF is password-protected. Upload an unlocked copy.")
        if doc.page_count == 0:
            raise PDFValidationError("The PDF contains no pages.")
        # Touch the first page to catch structurally broken documents early.
        try:
            doc.load_page(0)
        except Exception as exc:
            raise PDFValidationError("The PDF page tree is damaged and cannot be read.") from exc
        return doc.page_count
    finally:
        doc.close()


@lru_cache
def tesseract_status(tesseract_cmd: str | None = None) -> dict:
    """Detect whether Tesseract OCR is usable."""
    try:
        import pytesseract
    except ImportError:
        return {"available": False, "reason": "pytesseract is not installed"}
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif not shutil.which("tesseract"):
        return {"available": False, "reason": "tesseract binary not found on PATH (set TESSERACT_CMD)"}
    try:
        version = str(pytesseract.get_tesseract_version())
        return {"available": True, "version": version}
    except Exception as exc:
        return {"available": False, "reason": f"tesseract could not run: {exc}"}


@lru_cache
def _rapidocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR()


@lru_cache
def rapidocr_status() -> dict:
    """RapidOCR (pip-installable, models bundled) is used when Tesseract is not installed."""
    import importlib.util

    if importlib.util.find_spec("rapidocr") and importlib.util.find_spec("onnxruntime"):
        return {"available": True, "engine": "rapidocr"}
    return {"available": False, "reason": "rapidocr is not installed (pip install rapidocr onnxruntime)"}


def ocr_status(settings: Settings) -> dict:
    """Pick the OCR engine: Tesseract when installed, otherwise the bundled RapidOCR."""
    if not settings.ocr_enabled:
        return {"available": False, "reason": "OCR is disabled"}
    t = tesseract_status(settings.tesseract_cmd)
    if t.get("available"):
        return {**t, "engine": "tesseract"}
    r = rapidocr_status()
    if r.get("available"):
        return r
    return {"available": False, "reason": f"{t.get('reason')}; {r.get('reason')}"}


def _ocr_page(page: fitz.Page, settings: Settings, engine: str = "tesseract") -> str:
    if engine == "rapidocr":
        import numpy as np

        pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        res = _rapidocr_engine()(img)
        if not res.txts:
            return ""
        # group recognised boxes into lines (top-to-bottom, then left-to-right)
        items = sorted(zip(res.boxes, res.txts), key=lambda bt: (round(min(p[1] for p in bt[0]) / 12), min(p[0] for p in bt[0])))
        lines, cur, cur_y = [], [], None
        for box, txt in items:
            y = round(min(p[1] for p in box) / 12)
            if cur_y is not None and y != cur_y:
                lines.append(" ".join(cur))
                cur = []
            cur.append(txt)
            cur_y = y
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

    import pytesseract
    from PIL import Image

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    zoom = settings.ocr_dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=settings.ocr_language)


def _clean_page_text(text: str) -> str:
    # Keep line structure (headings matter); normalise trailing whitespace only.
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def extract_pdf(path: str, settings: Settings) -> ExtractionResult:
    """Extract text page by page. Raises PDFValidationError if the file cannot be opened."""
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PDFValidationError(f"The PDF is corrupted or cannot be opened ({exc.__class__.__name__}).") from exc

    with doc:
        if doc.needs_pass:
            raise PDFValidationError("The PDF is password-protected.")
        result = ExtractionResult(page_count=doc.page_count)
        if doc.page_count == 0:
            return result

        ocr_info = ocr_status(settings)
        needs_ocr: list[int] = []
        without_images = 0

        for idx in range(doc.page_count):
            page_no = idx + 1
            try:
                page = doc.load_page(idx)
                text = _clean_page_text(page.get_text("text", sort=True))
            except Exception as exc:
                log.warning("Text extraction failed on page %s of %s: %s", page_no, path, exc)
                text = ""
            meaningful = len("".join(text.split()))
            if meaningful >= settings.min_page_chars:
                result.pages.append(PageResult(page_no, text, len(text), "text", "ok"))
            else:
                result.pages.append(PageResult(page_no, text, len(text), "none", "empty"))
                try:
                    has_visuals = bool(page.get_images(full=False)) or bool(page.get_drawings())
                except Exception:
                    has_visuals = False
                if has_visuals:
                    needs_ocr.append(idx)          # a scanned or vector-rendered page with no text layer
                else:
                    without_images += 1            # blank, or the PDF's content is damaged
        result.unreadable_without_images = without_images

        if needs_ocr:
            if not settings.ocr_enabled:
                result.ocr_status = "disabled"
                result.ocr_message = f"{len(needs_ocr)} page(s) had no extractable text; OCR is disabled."
            elif not ocr_info.get("available"):
                result.ocr_status = "unavailable"
                result.ocr_message = (
                    f"{len(needs_ocr)} page(s) look scanned but OCR is unavailable: {ocr_info.get('reason')}."
                )
            else:
                used = 0
                for idx in needs_ocr:
                    page_no = idx + 1
                    try:
                        text = _clean_page_text(_ocr_page(doc.load_page(idx), settings, ocr_info.get("engine", "tesseract")))
                    except Exception as exc:
                        log.exception("OCR failed on page %s of %s", page_no, path)
                        result.pages[idx] = PageResult(page_no, "", 0, "ocr", "ocr_failed")
                        result.ocr_message = f"OCR failed on some pages: {exc}"
                        continue
                    if len("".join(text.split())) >= settings.min_page_chars:
                        result.pages[idx] = PageResult(page_no, text, len(text), "ocr", "ok")
                        used += 1
                    else:
                        result.pages[idx] = PageResult(page_no, text, len(text), "ocr", "empty")
                result.ocr_status = "used" if used else "failed"
                if not used and not result.ocr_message:
                    result.ocr_message = "OCR ran but found no readable text on the scanned pages."
        return result
