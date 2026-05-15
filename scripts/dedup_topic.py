"""Greedy near-duplicate dedup within a single topic.

For each cosine-similarity pair (a, b) above threshold, treat as edge in a graph.
Within each connected component, keep the artifact with the longest body, delete the rest.

Run with --dry-run first to see what would be deleted.
"""

import math
from collections import defaultdict
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import _parse_vec   # type: ignore


app = typer.Typer()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[rx] = ry


@app.command()
def main(
    topic: str = typer.Option(..., "--topic"),
    threshold: float = typer.Option(0.95, "--threshold"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    settings = get_settings()
    conn = connect(settings.database_url)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text AS id, artifact_type, title, body,
                       embedding::text AS embedding
                FROM semantic_items
                WHERE topic_id = %s AND embedding IS NOT NULL
                """,
                (topic,),
            )
            rows = list(cur.fetchall())

        typer.echo(f"Topic '{topic}': {len(rows)} artifacts with embeddings")

        # Parse vectors once
        parsed: list[dict] = []
        for r in rows:
            v = _parse_vec(r["embedding"])
            parsed.append({
                "id": r["id"],
                "artifact_type": r["artifact_type"],
                "title": r["title"],
                "body": r["body"] or "",
                "vec": v,
                "len": len(r["body"] or ""),
            })

        # Build edges
        uf = UnionFind()
        for p in parsed:
            uf.find(p["id"])

        pair_count = 0
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if parsed[i]["artifact_type"] != parsed[j]["artifact_type"]:
                    continue   # Only dedup within same artifact_type
                c = _cosine(parsed[i]["vec"], parsed[j]["vec"])
                if c >= threshold:
                    uf.union(parsed[i]["id"], parsed[j]["id"])
                    pair_count += 1

        # Group by root
        groups: dict[str, list[dict]] = defaultdict(list)
        for p in parsed:
            groups[uf.find(p["id"])].append(p)

        # Decide who to keep / delete
        to_delete: list[dict] = []
        kept_count = 0
        for root, members in groups.items():
            if len(members) == 1:
                kept_count += 1
                continue
            # Keep the one with the longest body; ties broken by earliest id
            members.sort(key=lambda m: (-m["len"], m["id"]))
            keeper = members[0]
            kept_count += 1
            for m in members[1:]:
                to_delete.append(m)

        typer.echo(f"  near-dup pairs (within same artifact_type): {pair_count}")
        typer.echo(f"  unique components: {len(groups)}")
        typer.echo(f"  would keep: {kept_count}")
        typer.echo(f"  would delete: {len(to_delete)}")

        if to_delete[:5]:
            typer.echo("\n  Sample deletions:")
            for m in to_delete[:5]:
                typer.echo(f"    [{m['artifact_type']}] '{m['title']}' (body_len={m['len']})")

        if dry_run:
            typer.echo("\nDRY RUN — no deletes performed.")
            return

        # Apply deletes
        ids_to_delete = [m["id"] for m in to_delete]
        if not ids_to_delete:
            typer.echo("\nNothing to delete.")
            return

        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM semantic_items WHERE id = ANY(%s::uuid[])",
                (ids_to_delete,),
            )
            deleted = cur.rowcount
        conn.commit()
        typer.echo(f"\nDeleted {deleted} artifacts.")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
