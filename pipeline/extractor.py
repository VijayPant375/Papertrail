"""
pipeline/extractor.py

Converts an uploaded PDF into structured text chunks (one chunk per page)
suitable for downstream LLM fact extraction.
"""

import re
import pdfplumber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """
    Strip leading/trailing whitespace, collapse 3+ consecutive newlines into 2,
    and remove null bytes from extracted page text.
    """
    if not raw:
        return ""
    # Remove null bytes
    text = raw.replace("\x00", "")
    # Collapse 3+ consecutive newlines into exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip overall leading/trailing whitespace
    return text.strip()


def _render_table(rows: list[list]) -> str | None:
    """
    Render a pdfplumber table (list of lists) as a pipe-delimited markdown-style
    string.  Returns None if every cell in the table is None or empty string so
    the caller can skip it.
    """
    if not rows:
        return None

    # Check whether all cells are None / empty
    all_empty = all(
        (cell is None or str(cell).strip() == "")
        for row in rows
        for cell in row
    )
    if all_empty:
        return None

    rendered_rows: list[str] = []
    for row in rows:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rendered_rows.append(" | ".join(cells))

    return "\n".join(rendered_rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_page_count(pdf_path: str) -> int:
    """Return the total number of pages in the PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_chunks(pdf_path: str) -> list[dict]:
    """
    Extract text and tables from every page of *pdf_path* and return a list of
    chunk dicts — one per page — with these keys:

        page_number  (int)        1-indexed page number
        text         (str)        cleaned page text
        tables       (list[str])  pipe-delimited table strings (may be empty)
        chunk_index  (int)        sequential 0-based index across all chunks

    Pages that yield no text and no tables are still included with a sentinel
    message so the pipeline can flag them as potential scan/image pages.
    """
    chunks: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for chunk_index, page in enumerate(pdf.pages):
            page_number = page.page_number  # pdfplumber uses 1-based page numbers

            # ---- Text -------------------------------------------------------
            raw_text = page.extract_text() or ""
            cleaned_text = _clean_text(raw_text)

            # ---- Tables -----------------------------------------------------
            raw_tables = page.extract_tables() or []
            tables: list[str] = []
            for raw_table in raw_tables:
                rendered = _render_table(raw_table)
                if rendered is not None:
                    tables.append(rendered)

            # ---- Fallback for image/scan pages ------------------------------
            if not cleaned_text and not tables:
                cleaned_text = "[no extractable text on this page]"

            chunks.append(
                {
                    "page_number": page_number,
                    "text": cleaned_text,
                    "tables": tables,
                    "chunk_index": chunk_index,
                }
            )

    return chunks
