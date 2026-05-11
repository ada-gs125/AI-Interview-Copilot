from io import BytesIO

import pdfplumber
from pypdf import PdfReader


def extract_resume_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    text = _extract_with_pdfplumber(pdf_bytes) or _extract_with_pypdf2(pdf_bytes)
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if len(normalized) < 50:
        raise ValueError("Could not extract enough text from the PDF resume.")

    return normalized


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return ""


def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

