"""Document parsing — ARCH-02 §8, M8.

Formats: PDF (text-based), DOCX, TXT, pasted text. Website pages are a
separate, later module (+$100 tier) — NOT in this Basic template.
Scanned/image-only PDFs are REFUSED with a written explanation, not
silently indexed as empty (ARCH-02 §3.4). No verified free reliable
Arabic OCR path exists; do not accept a scanned-PDF order on the
promise that OCR "will probably work".
"""

import hashlib
import os

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ParseRefusal(Exception):
    """A document refused at intake, with a reason for the buyer."""
    def __init__(self, reason, note):
        super().__init__(reason)
        self.reason = reason
        self.note = note


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def extract_text(path):
    """Extract text from one file. Returns (text, pages_or_None).

    Raises ParseRefusal for scanned/image-only PDFs and unsupported
    formats. Raises ValueError for unreadable files.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(), None
    raise ParseRefusal("unsupported_format",
                       f"Unsupported format {ext!r}. Supported: PDF, DOCX, "
                       f"TXT, pasted text.")


def _extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    text = "\n".join(pages_text)
    if not text.strip():
        raise ParseRefusal(
            "refused_scanned",
            "This PDF contains no extractable text (scanned/image-only). "
            "Text-based PDFs only — please supply a text version of the "
            "document. No OCR path is offered (ARCH-02 §3.4).")
    return text, len(reader.pages)


def _extract_docx(path):
    from docx import Document
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    # Tables: keep rows together so a price list stays one retrievable unit.
    for table in doc.tables:
        for row in table.rows:
            text += "\n" + " | ".join(c.text for c in row.cells)
    return text, None


def extract_pasted_text(text):
    """Pasted text supplied by the buyer (no file)."""
    if not text or not text.strip():
        raise ParseRefusal("empty_document", "Pasted text is empty.")
    return text.strip(), None
