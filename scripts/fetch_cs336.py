"""Fetch CS336 lecture .py files from GitHub and convert each to markdown under data/curriculum/."""

from pathlib import Path
import httpx
import typer

from learning_memory_os.ingestion.lecture_to_markdown import convert_lecture_py


app = typer.Typer()

REPO_RAW_BASE = "https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main"

LECTURE_MAP = {
    "lecture_01.py": "cs336_l01_overview.md",
    "lecture_02.py": "cs336_l02_resource_accounting.md",
    "lecture_06.py": "cs336_l06_kernels.md",
    "lecture_08.py": "cs336_l08_parallelism.md",
    "lecture_10.py": None,
    "lecture_12.py": "cs336_l12_evaluation.md",
    "lecture_13.py": "cs336_l13_data.md",
    "lecture_14.py": "cs336_l14_data.md",
    "lecture_17.py": "cs336_l17_rl_systems.md",
}


@app.command()
def main(
    out_dir: Path = typer.Option(Path("data/curriculum"), "--out-dir"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    skipped = 0
    failed = 0
    for lecture_file, md_name in LECTURE_MAP.items():
        if md_name is None:
            typer.echo(f"skip {lecture_file} (curated separately)")
            skipped += 1
            continue
        url = f"{REPO_RAW_BASE}/{lecture_file}"
        try:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            typer.echo(f"fetch failed for {lecture_file}: {e}", err=True)
            failed += 1
            continue
        md = convert_lecture_py(resp.text)
        target = out_dir / md_name
        target.write_text(md)
        typer.echo(f"wrote {target} ({len(md)} chars)")
        fetched += 1
    typer.echo(f"\nDone. Fetched: {fetched}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    app()
