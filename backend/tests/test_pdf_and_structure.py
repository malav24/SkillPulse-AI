import pytest

from app.config import get_settings
from app.services.pdf_processing import PDFValidationError, extract_pdf, ocr_status, rapidocr_status, validate_pdf_bytes
from tests.conftest import FIXTURES


# --- Unit-level PDF handling -------------------------------------------------------------------
def test_validate_rejects_invalid_files():
    with pytest.raises(PDFValidationError, match="empty"):
        validate_pdf_bytes(b"")
    with pytest.raises(PDFValidationError, match="signature"):
        validate_pdf_bytes((FIXTURES / "not_a_pdf.pdf").read_bytes())
    with pytest.raises(PDFValidationError):
        validate_pdf_bytes((FIXTURES / "corrupted.pdf").read_bytes())
    assert validate_pdf_bytes((FIXTURES / "sample_syllabus.pdf").read_bytes()) == 3


def test_text_extraction_preserves_pages():
    res = extract_pdf(str(FIXTURES / "sample_syllabus.pdf"), get_settings())
    assert res.page_count == 3 and res.method == "text" and res.empty_pages == []
    assert "Course Code: CS501" in res.pages[0].text
    assert "Unit I" in res.pages[1].text and res.pages[1].page_number == 2
    assert "List of Practicals" in res.pages[2].text and res.pages[2].page_number == 3


def test_blank_pdf_is_detected_as_empty():
    res = extract_pdf(str(FIXTURES / "blank.pdf"), get_settings())
    assert res.total_chars == 0 and res.empty_pages == [1]


@pytest.mark.skipif(not ocr_status(get_settings()).get("available"), reason="no OCR engine installed")
def test_ocr_fallback_reads_scanned_pdf():
    res = extract_pdf(str(FIXTURES / "scanned_syllabus.pdf"), get_settings())
    assert res.ocr_status == "used" and res.method == "ocr"
    assert "CS501" in res.pages[0].text


@pytest.mark.skipif(not rapidocr_status().get("available"), reason="RapidOCR not installed")
def test_rapidocr_reads_scanned_pdf_without_tesseract(monkeypatch):
    from app.services import pdf_processing

    monkeypatch.setattr(pdf_processing, "tesseract_status", lambda cmd=None: {"available": False, "reason": "absent"})
    assert pdf_processing.ocr_status(get_settings())["engine"] == "rapidocr"
    res = extract_pdf(str(FIXTURES / "scanned_syllabus.pdf"), get_settings())
    assert res.ocr_status == "used" and res.method == "ocr"
    text = res.pages[0].text
    assert "CS501" in text and "Machine Learning" in text


def test_ocr_unavailable_gives_clear_status():
    s = get_settings().model_copy(update={"ocr_enabled": False})
    res = extract_pdf(str(FIXTURES / "scanned_syllabus.pdf"), s)
    assert res.ocr_status == "disabled" and res.total_chars == 0
    assert "OCR is disabled" in res.ocr_message


# --- Upload API ----------------------------------------------------------------------------------




















REAL_SYLLABUS = FIXTURES.parent.parent.parent / "sample_data" / "JNTUH_R22_BTech_CSE_AIML_II_Year_Syllabus.pdf"


@pytest.mark.skipif(not REAL_SYLLABUS.exists(), reason="real JNTUH sample not present")
def test_real_multi_course_syllabus_structure():
    """Regression on a REAL syllabus (JNTUH R22 CSE AI&ML II Year): course headings, university, outcome intros."""
    from app.services.structure_parser import load_rules, parse_syllabus

    s = get_settings()
    res = extract_pdf(str(REAL_SYLLABUS), s)
    assert res.page_count == 29 and res.method == "text"
    parsed = parse_syllabus([(p.page_number, p.text) for p in res.pages if p.status == "ok"], load_rules(s.parsing_rules_path))
    assert parsed.metadata["university_name"]["value"] == "JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY HYDERABAD"
    assert parsed.metadata["course_code"]["value"] is None          # multi-course document: not guessed
    courses = [x.title for x in parsed.sections if x.section_type == "course"]
    assert len(courses) == 19 and "Data Structures" in courses and "Database Management Systems" in courses
    units = [x for x in parsed.sections if x.section_type == "unit"]
    assert all(u.course for u in units)
    assert not any(o.text.lower().startswith("after learning the contents") for o in parsed.outcomes)
    ds_units = [u for u in units if u.course == "Data Structures"]
    assert [u.unit_number for u in ds_units] == ["I", "II", "III", "IV", "V"]


def test_running_headers_and_footers_are_removed():
    from app.services.pipeline import strip_running_lines

    body = "\n".join(f"Topic line {i} about operating systems and scheduling" for i in range(12))
    pages = [(i, f"R22 B.Tech. CSE (AI and ML) Syllabus\nJNTU Hyderabad\n{body}\nUnit {i} content\nPage {i} of 6")
             for i in range(1, 7)]
    out = strip_running_lines(pages)
    assert out[0] == pages[0]                                   # title page kept intact
    for _, text in out[1:]:
        assert "Syllabus" not in text and "JNTU Hyderabad" not in text and "Page" not in text
        assert "Topic line 5 about operating systems" in text   # content stays
    assert strip_running_lines(pages[:2]) == pages[:2]          # too few pages to tell


def damaged_pdf_bytes() -> bytes:
    """A one-page PDF whose content stream is corrupt (claims Flate compression but holds garbage)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Course Title: Data Structures and Algorithms with plenty of text here")
    xref = page.get_contents()[0]
    doc.update_stream(xref, b"\x8b\x12garbage-not-deflate\x00\xff" * 20, compress=False)
    doc.xref_set_key(xref, "Filter", "/FlateDecode")
    return doc.tobytes()


def test_damaged_pdf_is_reported_as_damaged_not_scanned(client, tmp_path):
    p = tmp_path / "damaged.pdf"
    p.write_bytes(damaged_pdf_bytes())
    res = extract_pdf(str(p), get_settings())
    assert res.total_chars == 0 and res.unreadable_without_images == 1 and res.ocr_status == "not_needed"
