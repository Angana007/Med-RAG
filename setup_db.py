"""
Module: setup_db.py

One-shot data-initialization script — run this once before starting the
API server (and again any time the source *.json record files change):
    python setup_db.py

Thin convenience wrapper around database.py: creates the SQLite table if
it doesn't exist (init_db), then loads every *.json file in the current
directory into it (populate_db). Safe to re-run — populate_db uses
INSERT OR IGNORE keyed on (mrd_number, visit_id), so it won't duplicate
rows.

Note: this only sets up the SQL side. To also (re)build the FAISS vector
index used for semantic search, run `python embeddings.py` as well —
see the README's "Data Initialization" section for the full setup order.
"""

from database import init_db, populate_db

init_db()
populate_db()