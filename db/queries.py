"""
db/queries.py

High-level query helpers for the fact-knowledge-layer database.
Every function obtains its connection via get_db() from db/schema.py.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from db.schema import get_db


# ---------------------------------------------------------------------------
# documents table
# ---------------------------------------------------------------------------

def save_document(filename: str, page_count: int, metadata: dict) -> str:
    """
    Insert a new row into the *documents* table.

    Args:
        filename:   Original filename of the uploaded PDF.
        page_count: Total number of pages in the document.
        metadata:   Arbitrary dict of extra metadata (stored as JSON).

    Returns:
        The newly generated document id (UUID string).
    """
    doc_id = str(uuid.uuid4())
    upload_time = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata)

    conn = get_db()
    conn.execute(
        """
        INSERT INTO documents (id, filename, upload_time, page_count, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, filename, upload_time, page_count, metadata_json),
    )
    conn.close()

    return doc_id


def get_document(document_id: str) -> dict | None:
    """
    Fetch a single document row by *document_id*.

    Returns:
        A dict with document fields (metadata parsed back to a dict), or
        None if no row with that id exists.
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (document_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    doc = dict(row)
    doc["metadata"] = json.loads(doc["metadata"]) if doc["metadata"] else {}
    return doc


def list_documents() -> list[dict]:
    """
    Return all documents ordered by *upload_time* descending.

    Returns:
        List of document dicts (metadata parsed back to dicts).
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM documents ORDER BY upload_time DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    documents = []
    for row in rows:
        doc = dict(row)
        doc["metadata"] = json.loads(doc["metadata"]) if doc["metadata"] else {}
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# processing_log table
# ---------------------------------------------------------------------------

def log_stage(document_id: str, stage: str, status: str, message: str) -> None:
    """
    Append a row to *processing_log* to record the outcome of one pipeline
    stage for a document.

    Args:
        document_id: ID of the document being processed.
        stage:       Name of the pipeline stage (e.g. "extraction", "fact_finding").
        status:      Outcome string (e.g. "success", "error", "started").
        message:     Human-readable detail or error description.
    """
    log_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO processing_log (id, document_id, stage, status, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (log_id, document_id, stage, status, message, created_at),
    )
    conn.close()
