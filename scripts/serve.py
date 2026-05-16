"""Launch the Learning Memory OS web app: FastAPI + static frontend."""

import typer
import uvicorn

app = typer.Typer()


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    uvicorn.run(
        "learning_memory_os.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
