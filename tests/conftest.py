"""
Test bootstrap: builds the demo SQLite database if it's missing.

clinical_data.db is gitignored (generated data), so a fresh clone — and
every CI run — starts without it. This builds it from the committed
synthetic_patient_records.json in an isolated temp directory, so stray
JSON files in the repo root (e.g. eval_test_cases.json) are never
ingested as patient records.
"""

import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DB_PATH = os.path.join(REPO_ROOT, "clinical_data.db")
SOURCE_JSON = os.path.join(REPO_ROOT, "synthetic_patient_records.json")


def _build_demo_db() -> None:
    from database import init_db, populate_db

    original_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix="medrag_db_")
    try:
        shutil.copy(SOURCE_JSON, tmp_dir)
        os.chdir(tmp_dir)  # database.py resolves DB_NAME relative to cwd
        init_db()
        populate_db()
        shutil.move(os.path.join(tmp_dir, "clinical_data.db"), DB_PATH)
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if not os.path.exists(DB_PATH):
    _build_demo_db()
