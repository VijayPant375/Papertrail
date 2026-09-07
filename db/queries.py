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


# ---------------------------------------------------------------------------
# facts table
# ---------------------------------------------------------------------------

def save_fact(document_id: str, fact: dict, embedding: list[float]) -> str:
    """
    Insert one extracted fact into the *facts* table.

    The embedding list is serialised to raw bytes via numpy
    (float32, little-endian) for sqlite-vec compatibility.

    The *context* column stores a JSON string of {period, scope, unit}.

    Args:
        document_id: Parent document UUID.
        fact:        Dict produced by extract_facts_from_chunk (plus evidence_page).
        embedding:   List of floats from get_embedding().

    Returns:
        The newly generated fact id (UUID string).
    """
    import numpy as np  # local import keeps top-level startup fast

    fact_id = str(uuid.uuid4())

    context = json.dumps(
        {
            "period": fact.get("period"),
            "scope": fact.get("scope"),
            "unit": fact.get("unit"),
        }
    )

    embedding_bytes = (
        np.array(embedding, dtype=np.float32).tobytes() if embedding else b""
    )

    conn = get_db()
    conn.execute(
        """
        INSERT INTO facts (
            id, document_id, fact_type, subject, predicate,
            value, value_normalized, context, evidence_quote,
            evidence_page, confidence, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fact_id,
            document_id,
            fact.get("fact_type"),
            fact.get("subject"),
            fact.get("predicate"),
            fact.get("value"),
            fact.get("value_normalized"),      # may be None until normalisation step
            context,
            fact.get("evidence_quote"),
            fact.get("evidence_page"),
            fact.get("confidence"),
            embedding_bytes,
        ),
    )
    conn.close()

    return fact_id


def get_facts_for_document(document_id: str) -> list[dict]:
    """
    Return all facts belonging to *document_id* as a list of dicts, with the
    *context* column parsed back from JSON into a nested dict.

    The *embedding* column is NOT included — use get_all_facts_except_document
    when raw embedding bytes are required.
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, document_id, fact_type, subject, predicate,
               value, value_normalized, context, evidence_quote,
               evidence_page, confidence
        FROM facts
        WHERE document_id = ?
        """,
        (document_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    facts = []
    for row in rows:
        f = dict(row)
        f["context"] = json.loads(f["context"]) if f["context"] else {}
        facts.append(f)

    return facts


def get_all_facts_except_document(document_id: str) -> list[dict]:
    """
    Return all facts NOT belonging to *document_id*, including raw embedding
    bytes.  The comparator uses these bytes directly (numpy will decode them).

    The *context* column is left as a raw JSON string — callers should parse
    if needed.
    """
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, document_id, fact_type, subject, predicate,
               value, value_normalized, context, evidence_quote,
               evidence_page, confidence, embedding
        FROM facts
        WHERE document_id != ?
        """,
        (document_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# documents table — update helpers
# ---------------------------------------------------------------------------

def update_document_metadata(document_id: str, metadata: dict) -> None:
    """
    Overwrite the *metadata* column of an existing documents row.

    Args:
        document_id: UUID of the document to update.
        metadata:    New metadata dict (serialised to JSON before storage).
    """
    conn = get_db()
    conn.execute(
        "UPDATE documents SET metadata = ? WHERE id = ?",
        (json.dumps(metadata), document_id),
    )
    conn.close()
