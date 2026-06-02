"""Apply all SQL migrations in order. Idempotent — safe to re-run.

Every migration uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS, so this simply
applies each file in ``migrations/`` sorted by filename against the configured
database. Used both in dev and by the test suite to provision a fresh DB.

    uv run python -m scripts.migrate
    uv run python -m scripts.migrate --database-url postgresql://...
"""

from pathlib import Path

import psycopg
import typer

from learning_memory_os.config import get_settings

app = typer.Typer()

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def apply_migrations(database_url: str) -> list[str]:
    """Apply every migration file in sorted order. Returns the filenames applied."""
    files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    applied = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        for f in files:
            conn.execute(f.read_text())
            applied.append(f.name)
    return applied


@app.command()
def main(database_url: str = typer.Option(None, "--database-url")):
    url = database_url or get_settings().database_url
    applied = apply_migrations(url)
    typer.echo(f"Applied {len(applied)} migrations: {', '.join(applied)}")


if __name__ == "__main__":
    app()
