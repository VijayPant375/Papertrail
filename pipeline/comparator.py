"""
pipeline/comparator.py

Cross-document fact comparison using cosine-similarity candidate selection
(with a subject/fact_type fallback when embeddings are empty) followed by
an LLM-powered relationship classification pass.

Public API
----------
cosine_similarity(a, b)                    -> float
find_candidate_pairs(doc_a_id, doc_b_id)   -> list[tuple[dict, dict]]
compare_fact_pair(client, a, b)            -> dict | None
compare_document_pair(doc_a_id, doc_b_id)  -> int   (relationships saved)
trigger_comparison(document_id)            -> None
"""

import json
import logging

import numpy as np
from google import genai
from google.genai import types

from pipeline.prompts import RELATIONSHIP_PROMPT
from db.queries import (
    get_all_facts_except_document,
    get_document,
    get_facts_for_document,
    list_documents,
    log_stage,
    save_relationship,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.75
_MAX_CANDIDATE_PAIRS = 200   # hard cap for both embedding and fallback paths
_FALLBACK_MIN_PAIRS = 1      # if embedding yields fewer than this, run fallback

# Gemini models used for relationship classification (with automatic fallback)
_COMPARISON_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
]

_VALID_RELATIONSHIP_TYPES = {"corroborates", "contradicts", "reconcilable", "unrelated"}


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: bytes, b: bytes) -> float:
    """
    Compute cosine similarity between two float32 embedding byte-blobs.

    Args:
        a: Raw bytes of a numpy float32 embedding (as stored in the DB).
        b: Raw bytes of a numpy float32 embedding (as stored in the DB).

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero-length
        or the byte blobs are empty / mismatched.
    """
    if not a or not b:
        return 0.0

    try:
        vec_a = np.frombuffer(a, dtype=np.float32)
        vec_b = np.frombuffer(b, dtype=np.float32)
    except Exception:
        return 0.0

    if vec_a.shape != vec_b.shape or vec_a.size == 0:
        return 0.0

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Candidate pair selection
# ---------------------------------------------------------------------------

def find_candidate_pairs(
    doc_a_id: str,
    doc_b_id: str,
) -> list[tuple[dict, dict]]:
    """
    Find candidate fact pairs between *doc_a_id* and *doc_b_id* for LLM
    relationship classification.

    Primary path — cosine similarity on stored embeddings
    -------------------------------------------------------
    1. Fetch facts (with embeddings) for both documents.
    2. Compute cosine similarity for every cross-document pair.
    3. Keep pairs with similarity >= _SIMILARITY_THRESHOLD.
    4. Sort descending and cap at _MAX_CANDIDATE_PAIRS.

    Fallback path — structural matching (used when embeddings are empty/missing)
    ----------------------------------------------------------------------------
    If the primary path yields fewer than _FALLBACK_MIN_PAIRS, fall back to
    matching facts where:
        subject LIKE %other_subject%  OR  fact_type = other_fact_type
    across the two documents.  Cap at _MAX_CANDIDATE_PAIRS.

    Returns:
        List of (fact_a, fact_b) tuples, fact_a from *doc_a_id*.
    """
    facts_a = _get_facts_with_embeddings(doc_a_id)
    facts_b = _get_facts_with_embeddings(doc_b_id)

    if not facts_a or not facts_b:
        return []

    # ---- Primary: embedding cosine similarity --------------------------------
    scored: list[tuple[float, dict, dict]] = []

    for fact_a in facts_a:
        emb_a = fact_a.get("embedding") or b""
        for fact_b in facts_b:
            emb_b = fact_b.get("embedding") or b""
            sim = cosine_similarity(emb_a, emb_b)
            if sim >= _SIMILARITY_THRESHOLD:
                scored.append((sim, fact_a, fact_b))

    if len(scored) >= _FALLBACK_MIN_PAIRS:
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(fa, fb) for _, fa, fb in scored[:_MAX_CANDIDATE_PAIRS]]

    # ---- Fallback: subject / fact_type structural matching ------------------
    logger.info(
        "find_candidate_pairs: embedding path found %d pairs for (%s, %s); "
        "running subject/fact_type fallback.",
        len(scored),
        doc_a_id,
        doc_b_id,
    )

    seen: set[tuple[str, str]] = set()
    fallback: list[tuple[dict, dict]] = []

    for fact_a in facts_a:
        subj_a = (fact_a.get("subject") or "").lower()
        type_a = (fact_a.get("fact_type") or "").lower()

        for fact_b in facts_b:
            pair_key = (fact_a["id"], fact_b["id"])
            if pair_key in seen:
                continue

            subj_b = (fact_b.get("subject") or "").lower()
            type_b = (fact_b.get("fact_type") or "").lower()

            # Match if subjects overlap OR fact_types are identical
            subject_overlap = (
                subj_a and subj_b and (subj_a in subj_b or subj_b in subj_a)
            )
            type_match = type_a and type_b and type_a == type_b

            if subject_overlap or type_match:
                seen.add(pair_key)
                fallback.append((fact_a, fact_b))
                if len(fallback) >= _MAX_CANDIDATE_PAIRS:
                    return fallback

    return fallback


def _get_facts_with_embeddings(document_id: str) -> list[dict]:
    """
    Internal helper: return facts for *document_id* including raw embedding bytes.
    Reuses the get_all_facts_except_document query pattern but inverted — we
    query facts WHERE document_id = ? and include the embedding column.
    """
    import sqlite3
    from db.schema import get_db

    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT id, document_id, fact_type, subject, predicate,
               value, value_normalized, context, evidence_quote,
               evidence_page, confidence, embedding
        FROM facts
        WHERE document_id = ?
        """,
        (document_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# LLM comparison
# ---------------------------------------------------------------------------

def compare_fact_pair(
    client: genai.Client,
    fact_a: dict,
    fact_b: dict,
) -> dict | None:
    """
    Use Gemini to classify the relationship between *fact_a* and *fact_b*.

    Each fact dict is enriched with its source document's filename before
    being serialised into the prompt so the model has document provenance.

    Args:
        client: Initialised google.genai.Client.
        fact_a: Fact dict from the new document (includes document_id).
        fact_b: Fact dict from another document (includes document_id).

    Returns:
        A dict with keys relationship_type, confidence, explanation — or
        None if the relationship is "unrelated" or the LLM call fails.
    """
    # Enrich both facts with their source document filename
    def _enrich(fact: dict) -> dict:
        doc = get_document(fact.get("document_id", ""))
        source_filename = doc["filename"] if doc else "unknown"
        # Build a clean, minimal representation for the prompt
        return {
            "source_document": source_filename,
            "fact_type": fact.get("fact_type"),
            "subject": fact.get("subject"),
            "predicate": fact.get("predicate"),
            "value": fact.get("value"),
            "context": (
                json.loads(fact["context"])
                if isinstance(fact.get("context"), str) and fact["context"]
                else fact.get("context") or {}
            ),
            "evidence_quote": fact.get("evidence_quote"),
        }

    fact_a_payload = _enrich(fact_a)
    fact_b_payload = _enrich(fact_b)

    prompt = RELATIONSHIP_PROMPT.format(
        fact_a_json=json.dumps(fact_a_payload, indent=2, ensure_ascii=False),
        fact_b_json=json.dumps(fact_b_payload, indent=2, ensure_ascii=False),
    )

    result = None
    for model_name in _COMPARISON_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text.strip())
            break
        except json.JSONDecodeError as exc:
            logger.warning(
                "compare_fact_pair JSON parse error with %s: %s — raw: %r",
                model_name,
                exc,
                getattr(response, "text", ""),
            )
            return None
        except Exception as exc:
            logger.info(
                "Model %s failed during comparison (%s), trying fallback.",
                model_name,
                exc,
            )
            continue

    if not result:
        logger.warning("compare_fact_pair failed with all comparison models.")
        return None

    # Validate response shape
    rel_type = result.get("relationship_type", "").lower()
    if rel_type not in _VALID_RELATIONSHIP_TYPES:
        logger.warning("Unexpected relationship_type %r — discarding.", rel_type)
        return None

    if rel_type == "unrelated":
        return None  # Not worth storing

    return {
        "relationship_type": rel_type,
        "confidence": float(result.get("confidence", 0.5)),
        "explanation": result.get("explanation", ""),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare_document_pair(
    doc_a_id: str,
    doc_b_id: str,
    client: "genai.Client | None" = None,
) -> int:
    """
    Run the full candidate-selection → LLM classification pipeline for a
    single pair of documents and persist the results.

    Args:
        doc_a_id: UUID of the first document.
        doc_b_id: UUID of the second document.
        client:   Optional pre-initialised genai.Client; created from
                  GEMINI_API_KEY env-var if not supplied.

    Returns:
        Number of relationships saved.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("compare_document_pair: GEMINI_API_KEY not set.")
            return 0
        client = genai.Client(api_key=api_key)

    candidate_pairs = find_candidate_pairs(doc_a_id, doc_b_id)
    if not candidate_pairs:
        logger.info(
            "compare_document_pair: no candidate pairs for (%s, %s).",
            doc_a_id,
            doc_b_id,
        )
        return 0

    logger.info(
        "compare_document_pair: %d candidate pairs for (%s, %s).",
        len(candidate_pairs),
        doc_a_id,
        doc_b_id,
    )

    relationships_saved = 0
    for fact_a, fact_b in candidate_pairs:
        result = compare_fact_pair(client, fact_a, fact_b)
        if result is None:
            continue

        try:
            save_relationship(
                fact_id_a=fact_a["id"],
                fact_id_b=fact_b["id"],
                relationship_type=result["relationship_type"],
                explanation=result["explanation"],
                confidence=result["confidence"],
            )
            relationships_saved += 1
        except Exception as exc:
            logger.warning(
                "Failed to save relationship (%s, %s): %s",
                fact_a["id"],
                fact_b["id"],
                exc,
            )

    return relationships_saved


def trigger_comparison(document_id: str) -> None:
    """
    Orchestrate cross-document fact comparison for *document_id*.

    Called automatically at the end of process_document() in fact_finder.py.
    Compares the new document against **all** other documents already in the
    DB (not just ones uploaded before it), so uploading doc-2 always triggers
    a comparison with doc-1 regardless of insertion order.

    Steps
    -----
    1. Collect every document id except *document_id*.
    2. If none exist (first document), log and return early.
    3. Set up Gemini client.
    4. For each other document, run compare_document_pair().
    5. Log aggregate completion.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    log_stage(document_id, "comparison", "started", "Searching for candidate fact pairs.")

    # Collect all other document ids
    all_docs = list_documents()
    other_doc_ids = [d["id"] for d in all_docs if d["id"] != document_id]

    if not other_doc_ids:
        log_stage(
            document_id,
            "comparison",
            "skipped",
            "No other documents in the DB — this is the first document.",
        )
        return

    # Set up Gemini client once, reuse across document pairs
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_stage(document_id, "comparison", "error", "GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=api_key)

    total_relationships = 0
    total_pairs = 0

    for other_id in other_doc_ids:
        saved = compare_document_pair(document_id, other_id, client=client)
        total_relationships += saved
        # Re-fetch candidate count for logging accuracy is not critical; just note we ran
        total_pairs += 1

    log_stage(
        document_id,
        "comparison",
        "completed",
        f"Compared against {total_pairs} other document(s); "
        f"saved {total_relationships} relationship(s).",
    )
