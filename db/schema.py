import sqlite3
import os
import uuid
import sqlite_vec
from dotenv import load_dotenv

load_dotenv()


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Opens a SQLite connection, loads the sqlite-vec extension, and creates
    all four tables if they don't already exist.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        The open connection (autocommit mode, any-thread safe).
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)

    # Extension loading is disabled by default in Python's sqlite3 — enable it
    # briefly, load sqlite-vec, then lock it back down.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        -- ----------------------------------------------------------------
        -- documents: one row per uploaded PDF / source file
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            filename    TEXT,
            upload_time TEXT,
            page_count  INTEGER,
            metadata    TEXT   -- JSON string
        );

        -- ----------------------------------------------------------------
        -- facts: structured facts extracted from documents
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS facts (
            id               TEXT PRIMARY KEY,
            document_id      TEXT,
            fact_type        TEXT,
            subject          TEXT,
            predicate        TEXT,
            value            TEXT,
            value_normalized TEXT,
            context          TEXT,   -- JSON string: {period, scope, unit}
            evidence_quote   TEXT,
            evidence_page    INTEGER,
            confidence       REAL,
            embedding        BLOB    -- sqlite-vec vector blob
        );

        -- ----------------------------------------------------------------
        -- relationships: pairwise fact comparisons
        -- relationship_type: corroborates | contradicts | reconcilable
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS relationships (
            id                TEXT PRIMARY KEY,
            fact_id_a         TEXT,
            fact_id_b         TEXT,
            relationship_type TEXT,
            explanation       TEXT,
            confidence        REAL,
            created_at        TEXT
        );

        -- ----------------------------------------------------------------
        -- processing_log: per-document pipeline stage tracking
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS processing_log (
            id          TEXT PRIMARY KEY,
            document_id TEXT,
            stage       TEXT,
            status      TEXT,
            message     TEXT,
            created_at  TEXT
        );
    """)

    return conn


def get_db() -> sqlite3.Connection:
    """
    Returns a connection to the database whose path is configured in the
    DB_PATH environment variable (defaults to 'facts.db').

    The sqlite-vec extension is loaded on every new connection.
    """
    db_path = os.getenv("DB_PATH", "facts.db")
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
