"""PDF resume text extraction with pdfplumber primary and pypdf fallback."""

import logging
from io import BytesIO

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_resume_text(pdf_bytes: bytes) -> str:
    # Normalize extracted text and reject empty/scanned PDFs early.
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    text = _extract_with_pdfplumber(pdf_bytes) or _extract_with_pypdf2(pdf_bytes)
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if len(normalized) < 50:
        raise ValueError("Could not extract enough text from the PDF resume.")

    return normalized


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    # pdfplumber generally handles text PDFs better, so try it first.
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                logger.info("pdf_parser=pdfplumber pages=%d chars=%d", len(pdf.pages), len(text))
            return text
    except Exception as e:
        logger.warning("pdf_parser pdfplumber failed, trying pypdf: %s", e)
        return ""


def _extract_with_pypdf2(pdf_bytes: bytes) -> str:
    # Fallback parser keeps uploads working when pdfplumber cannot read a file.
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            logger.info("pdf_parser=pypdf pages=%d chars=%d", len(reader.pages), len(text))
        return text
    except Exception as e:
        logger.error("pdf_parser pypdf also failed: %s", e)
        return ""
