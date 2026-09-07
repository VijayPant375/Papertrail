"""
pipeline/comparator.py

Cross-document fact comparison using cosine-similarity candidate selection
followed by an LLM-powered relationship classification pass.

Public API
----------
cosine_similarity(a, b)              -> float
find_candidate_pairs(document_id)    -> list[tuple[dict, dict]]
compare_fact_pair(client, a, b)      -> dict | None
trigger_comparison(document_id)      -> None
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
    log_stage,
    save_relationship,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.75
_MAX_CANDIDATE_PAIRS = 100

# Prefer the more capable model; fall back to flash if not accessible
_COMPARISON_MODEL_PRIMARY = "gemini-1.5-pro"
_COMPARISON_MODEL_FALLBACK = "gemini-2.0-flash"

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

def find_candidate_pairs(document_id: str) -> list[tuple[dict, dict]]:
    """
    Find semantically similar fact pairs between the new document and all
    previously stored facts using cosine similarity on stored embeddings.

    Steps
    -----
    1. Fetch all facts for *document_id* (these are the "new" facts).
    2. Fetch all facts from every other document (the "existing" facts).
    3. Compute cosine similarity for every (new, existing) pair.
    4. Keep only pairs with similarity >= _SIMILARITY_THRESHOLD.
    5. Sort descending by similarity and cap at _MAX_CANDIDATE_PAIRS.

    Returns:
        List of (fact_a, fact_b) tuples where fact_a is from *document_id*.
    """
    new_facts = get_facts_for_document(document_id)
    # get_facts_for_document strips embeddings — we need them, so fetch again
    # via get_all_facts_except_document for the "other" side, and refetch new
    # facts with embeddings directly
    new_facts_with_emb = _get_facts_with_embeddings(document_id)
    other_facts = get_all_facts_except_document(document_id)

    if not new_facts_with_emb or not other_facts:
        return []

    scored: list[tuple[float, dict, dict]] = []

    for fact_a in new_facts_with_emb:
        emb_a = fact_a.get("embedding") or b""
        for fact_b in other_facts:
            emb_b = fact_b.get("embedding") or b""
            sim = cosine_similarity(emb_a, emb_b)
            if sim >= _SIMILARITY_THRESHOLD:
                scored.append((sim, fact_a, fact_b))

    # Sort highest similarity first, cap at limit
    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[:_MAX_CANDIDATE_PAIRS]

    return [(fact_a, fact_b) for _, fact_a, fact_b in scored]


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

    # Try primary model, fall back to flash on error
    for model_name in (_COMPARISON_MODEL_PRIMARY, _COMPARISON_MODEL_FALLBACK):
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
            if model_name == _COMPARISON_MODEL_PRIMARY:
                logger.info(
                    "Primary model %s unavailable (%s), trying fallback.",
                    model_name,
                    exc,
                )
                continue
            logger.warning("compare_fact_pair failed with both models: %s", exc)
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

def trigger_comparison(document_id: str) -> None:
    """
    Orchestrate cross-document fact comparison for *document_id*.

    Called automatically at the end of process_document() in fact_finder.py.

    Steps
    -----
    1. Find candidate fact pairs by vector similarity.
    2. If none exist (first document), log and return early.
    3. Set up Gemini client.
    4. For each pair, classify the relationship via LLM.
    5. Save non-unrelated relationships to the DB.
    6. Log completion with relationship count.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    log_stage(document_id, "comparison", "started", "Searching for candidate fact pairs.")

    candidate_pairs = find_candidate_pairs(document_id)

    if not candidate_pairs:
        log_stage(
            document_id,
            "comparison",
            "skipped",
            "No prior documents to compare — this is the first document or no similar facts found.",
        )
        return

    # Set up Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_stage(document_id, "comparison", "error", "GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=api_key)

    relationships_saved = 0

    for fact_a, fact_b in candidate_pairs:
        result = compare_fact_pair(client, fact_a, fact_b)
        if result is None:
            continue  # unrelated or failed — skip

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
                "Failed to save relationship between %s and %s: %s",
                fact_a["id"],
                fact_b["id"],
                exc,
            )

    log_stage(
        document_id,
        "comparison",
        "completed",
        f"Found and saved {relationships_saved} relationships from "
        f"{len(candidate_pairs)} candidate pairs.",
    )
