from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from app.services.pdf_parser import extract_resume_text


def _pdf_bytes(text: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, text)
    pdf.save()
    return buffer.getvalue()


def test_extract_resume_text_from_text_pdf():
    pdf = _pdf_bytes(
        "Python FastAPI backend engineer with SQL APIs and structured LLM output experience."
    )

    text = extract_resume_text(pdf)

    assert "Python FastAPI backend engineer" in text
    assert "structured LLM output" in text


def test_extract_resume_text_rejects_empty_upload():
    with pytest.raises(ValueError, match="empty"):
        extract_resume_text(b"")


def test_extract_resume_text_rejects_unparseable_pdf_bytes():
    with pytest.raises(ValueError, match="Could not extract enough text"):
        extract_resume_text(b"not a real pdf")
