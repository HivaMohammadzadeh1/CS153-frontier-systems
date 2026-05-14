"""Launch a single LoRA fine-tune by size id from config/router_sizes.yaml."""

from pathlib import Path
import yaml
import typer

from learning_memory_os.router.finetune import (
    RouterFineTuneConfig, finetune,
)


app = typer.Typer()


@app.command()
def main(
    size_id: str = typer.Option(..., "--size"),
    trajectories: Path = typer.Option(Path("data/trajectories/main.jsonl"), "--trajectories"),
    out: Path = typer.Option(Path("data/router_checkpoints"), "--out"),
    epochs: int = typer.Option(2, "--epochs"),
):
    cfg_all = yaml.safe_load(Path("config/router_sizes.yaml").read_text())
    sizes = {s["id"]: s for s in cfg_all["sizes"]}
    if size_id not in sizes:
        typer.echo(f"Unknown size: {size_id}", err=True)
        raise typer.Exit(2)
    s = sizes[size_id]
    cfg = RouterFineTuneConfig(
        hf_model=s["hf_model"],
        lora_r=s["lora_r"],
        lora_alpha=s["lora_alpha"],
        batch_size=s["batch_size"],
        max_seq_len=s["max_seq_len"],
        use_4bit_base=s["use_4bit_base"],
        epochs=epochs,
    )
    output_dir = out / size_id
    adapter_path = finetune(cfg, trajectories_path=trajectories, output_dir=output_dir)
    typer.echo(f"Done. Adapter at: {adapter_path}")


if __name__ == "__main__":
    app()
