"""
main.py

FastAPI application entry point for the fact-knowledge-layer / Papertrail project.

Startup:
  - Loads .env
  - Initialises the SQLite database (creates tables, loads sqlite-vec)
  - Mounts ui/ as static files

Routes:
  POST /upload
  GET  /documents
  GET  /documents/{document_id}
  GET  /documents/{document_id}/relationships
  GET  /relationships
  GET  /health
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load environment variables from .env before anything else touches os.getenv
load_dotenv()

from db.schema import init_db
from db.queries import (
    get_all_relationships,
    get_document,
    get_facts_for_document,
    get_relationships_for_document,
    list_documents,
)
from pipeline.fact_finder import process_document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and again at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    db_path = os.getenv("DB_PATH", "facts.db")
    logger.info("Initialising database at %s", db_path)
    init_db(db_path)
    logger.info("Database ready.")
    yield
    # Nothing to tear down for SQLite
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Papertrail — Fact Knowledge Layer",
    description=(
        "Upload financial/economic PDFs and automatically extract, embed, "
        "and cross-reference structured facts across documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Serve everything under ui/ at /ui (CSS, JS, assets)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the frontend SPA."""
    return FileResponse("ui/index.html")


@app.get("/health", tags=["Meta"])
async def health():
    """Simple liveness check."""
    return {"status": "ok"}


# ---- Upload ----------------------------------------------------------------

@app.post("/upload", tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, run the full extraction + comparison pipeline,
    and return the new document_id.

    The pipeline (PDF → chunks → LLM facts → embeddings → relationships)
    can take 30–120 s for a large document and is executed in a thread pool
    via asyncio.to_thread() to avoid blocking the event loop.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Write upload to a temp file so pdfplumber can open it by path
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        logger.info("Processing upload: %s → %s", file.filename, tmp_path)

        document_id = await asyncio.to_thread(
            process_document, tmp_path, file.filename
        )

        logger.info("Processing complete. document_id=%s", document_id)
        return {"document_id": document_id, "message": "Processing complete"}

    except Exception as exc:
        logger.exception("Upload processing failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        # Always remove the temp file, even on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


# ---- Documents -------------------------------------------------------------

@app.get("/documents", tags=["Documents"])
async def get_documents():
    """Return all documents ordered by upload time (newest first)."""
    return list_documents()


@app.get("/documents/{document_id}", tags=["Documents"])
async def get_document_detail(document_id: str):
    """
    Return a single document with its extracted facts embedded under the
    'facts' key.  Returns 404 if the document_id is not found.
    """
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    doc["facts"] = get_facts_for_document(document_id)
    return doc


# ---- Relationships ---------------------------------------------------------

@app.get("/documents/{document_id}/relationships", tags=["Relationships"])
async def get_document_relationships(document_id: str):
    """
    Return all cross-document relationships where at least one fact belongs
    to the given document.
    """
    # Validate document exists first
    if get_document(document_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return get_relationships_for_document(document_id)


@app.get("/relationships", tags=["Relationships"])
async def get_relationships(
    type: str | None = Query(
        default=None,
        description="Filter by relationship type: corroborates | contradicts | reconcilable",
    ),
):
    """
    Return all cross-document relationships.

    Optionally filter by type, e.g. GET /relationships?type=contradicts
    """
    relationships = get_all_relationships()

    if type is not None:
        valid_types = {"corroborates", "contradicts", "reconcilable"}
        if type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type '{type}'. Must be one of: {', '.join(sorted(valid_types))}",
            )
        relationships = [r for r in relationships if r["relationship_type"] == type]

    return relationships
