"""
pipeline/fact_finder.py

LLM-powered fact extraction pipeline using the Google Gemini API.

Public API
----------
setup_gemini()                          -> genai.Client
extract_document_metadata(client, text) -> dict
extract_facts_from_chunk(client, chunk, document_name) -> list[dict]
get_embedding(client, text)             -> list[float]
process_document(pdf_path, filename)    -> str  (document_id)
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pipeline.extractor import extract_chunks, get_page_count
from pipeline.prompts import DOCUMENT_METADATA_PROMPT, FACT_EXTRACTION_PROMPT
from pipeline.comparator import trigger_comparison
from db.queries import (
    get_facts_for_document,
    log_stage,
    save_document,
    save_fact,
    update_document_metadata,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

_GENERATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
_EMBEDDING_MODELS = [
    "gemini-embedding-002",
]


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """
    Generate a semantic similarity embedding for *text*.

    The caller should pass subject + " " + predicate + " " + value
    as the input text.

    Returns:
        A Python list of floats representing the embedding vector.
    """
    for model_name in _EMBEDDING_MODELS:
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                ),
            )
            return response.embeddings[0].values
        except Exception as exc:
            logger.warning("get_embedding failed with %s: %s", model_name, exc)
            continue

    return []


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def setup_gemini() -> genai.Client:
    """
    Read GEMINI_API_KEY from environment and return an initialised
    google.genai.Client.

    Raises:
        ValueError: if GEMINI_API_KEY is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_document_metadata(client: genai.Client, text: str) -> dict:
    """
    Send the first ~2 pages of *text* to Gemini and parse the returned JSON
    into a metadata dict.

    On any failure (network, parse error, etc.) a safe default dict is
    returned so the pipeline can continue.
    """
    prompt = DOCUMENT_METADATA_PROMPT.format(text=text)

    for model_name in _GENERATION_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("extract_document_metadata failed with %s: %s", model_name, exc)
            continue

    return {
        "title": "unknown",
        "document_type": "unknown",
        "organization": "unknown",
        "period_covered": None,
        "publication_date": None,
    }


# ---------------------------------------------------------------------------
# Per-chunk fact extraction
# ---------------------------------------------------------------------------

def extract_facts_from_chunk(
    client: genai.Client,
    chunk: dict,
    document_name: str,
) -> list[dict]:
    """
    Extract structured facts from a single page *chunk* using Gemini.

    Each returned fact dict includes all keys from the LLM response plus:
        evidence_page (int) — copied from chunk["page_number"]

    On JSON parse failure the raw response is logged and an empty list is
    returned so the pipeline keeps running.
    """
    # Combine page text and any table content into one block
    table_block = ""
    if chunk.get("tables"):
        table_block = "\n\nTables on this page:\n" + "\n\n".join(chunk["tables"])

    full_text = chunk["text"] + table_block

    prompt = FACT_EXTRACTION_PROMPT.format(
        document_name=document_name,
        page_number=chunk["page_number"],
        text=full_text,
    )

    for model_name in _GENERATION_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            facts = json.loads(raw)

            if not isinstance(facts, list):
                logger.warning(
                    "Chunk %d: expected JSON array, got %s — skipping.",
                    chunk["page_number"],
                    type(facts).__name__,
                )
                return []

            # Stamp each fact with the source page
            for fact in facts:
                fact["evidence_page"] = chunk["page_number"]

            return facts

        except Exception as exc:
            logger.warning(
                "Chunk %d: Gemini call failed with %s: %s",
                chunk["page_number"],
                model_name,
                exc,
            )
            continue

    return []




# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_document(pdf_path: str, filename: str) -> str:
    """
    Full pipeline orchestration for a single PDF document.

    Steps
    -----
    1. Extract chunks and page count from the PDF.
    2. Save a documents row and get back a document_id.
    3. Log extraction started.
    4. Extract document-level metadata from the first 2 pages.
    5. Update the documents row with the metadata.
    6. For every chunk: extract facts → embed each fact → save to DB.
    7. Log extraction completed with fact count.
    8. Trigger cross-document comparison.

    Args:
        pdf_path: Absolute or relative path to the PDF file on disk.
        filename: Original filename to store in the DB.

    Returns:
        The document_id string (UUID) for the newly created document row.
    """
    client = setup_gemini()

    # 1. Extract PDF content
    page_count = get_page_count(pdf_path)
    chunks = extract_chunks(pdf_path)

    # 2. Create document record (metadata will be filled in step 4)
    document_id = save_document(
        filename=filename,
        page_count=page_count,
        metadata={},
    )

    # 3. Log start
    log_stage(document_id, "extraction", "started", f"Processing {page_count} pages.")

    # 4. Extract document-level metadata from first 2 pages
    first_two_text = " ".join(
        c["text"] for c in chunks[:2] if c["text"] != "[no extractable text on this page]"
    )
    metadata = extract_document_metadata(client, first_two_text)

    # 5. Persist metadata
    update_document_metadata(document_id, metadata)

    # 6. Extract + embed + save facts, chunk by chunk
    total_facts_saved = 0

    for chunk in chunks:
        facts = extract_facts_from_chunk(client, chunk, document_name=filename)

        for fact in facts:
            # Build embedding input text
            embed_text = " ".join(
                filter(None, [fact.get("subject"), fact.get("predicate"), fact.get("value")])
            )
            try:
                embedding = get_embedding(client, embed_text)
            except Exception as exc:
                logger.warning(
                    "Embedding failed for fact (page %d, subject=%r): %s",
                    chunk["page_number"],
                    fact.get("subject"),
                    exc,
                )
                embedding = []

            save_fact(document_id=document_id, fact=fact, embedding=embedding)
            total_facts_saved += 1

    # 7. Log completion
    log_stage(
        document_id,
        "extraction",
        "completed",
        f"Saved {total_facts_saved} facts from {page_count} pages.",
    )

    # 8. Trigger cross-document comparison
    trigger_comparison(document_id)

    return document_id
