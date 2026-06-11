import os
import shutil
import sqlite3
from pathlib import Path

_FIXTURE_SQL = Path(__file__).parent / "dexter_index_dump.sql"


def create_dexter_index(repo_path: Path) -> Path:
    """
    Create a Dexter index (.dexter/dexter.db) for the Elixir test repository from the
    committed SQL dump (so that tests do not require the dexter binary).

    The dump was produced by running ``dexter init`` in the test repository and dumping
    the resulting database with the repository root replaced by ``/dexter-indexed-root``
    (which also exercises the index-root remapping logic of the reader) and file mtimes
    maxed out (to avoid staleness warnings). To regenerate it after changing the test
    repository, run ``dexter init`` there and re-apply these substitutions.
    """
    dexter_dir = repo_path / ".dexter"
    dexter_dir.mkdir(exist_ok=True)
    db_path = dexter_dir / "dexter.db"
    if db_path.exists():
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_FIXTURE_SQL.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
    return db_path


def remove_dexter_index(repo_path: Path) -> None:
    shutil.rmtree(repo_path / ".dexter", ignore_errors=True)
