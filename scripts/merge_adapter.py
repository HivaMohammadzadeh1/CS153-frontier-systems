"""Merge a LoRA adapter into its base model, producing a standalone model dir.

A merged model needs no PEFT/adapter at serve time — just `vllm serve <dir>` or
`AutoModelForCausalLM.from_pretrained(<dir>)`. Optionally push to the HF Hub.

Examples:
  # by size id (uses config/router_sizes.yaml + local adapter)
  python -m scripts.merge_adapter --size qwen2_5_7b --out data/merged/qwen2_5_7b

  # explicit base + adapter (adapter can be a local dir OR an HF repo id)
  python -m scripts.merge_adapter \
    --base Qwen/Qwen2.5-7B-Instruct \
    --adapter hivamoh/lmos-router-qwen2_5_7b \
    --out /opt/model --push-to hivamoh/lmos-router-qwen2_5_7b-merged
"""
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = typer.Typer()


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


@app.command()
def main(
    size: Optional[str] = typer.Option(None, "--size", help="id from config/router_sizes.yaml"),
    base: Optional[str] = typer.Option(None, "--base", help="base HF model id"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="adapter dir or HF repo id"),
    out: Path = typer.Option(..., "--out", help="output dir for merged model"),
    push_to: Optional[str] = typer.Option(None, "--push-to", help="HF repo to upload merged model to"),
    private: bool = typer.Option(True, "--private/--public"),
):
    if size:
        sizes = {s["id"]: s for s in yaml.safe_load(Path("config/router_sizes.yaml").read_text())["sizes"]}
        if size not in sizes:
            typer.echo(f"unknown size: {size}", err=True)
            raise typer.Exit(2)
        base = base or sizes[size]["hf_model"]
        adapter = adapter or f"data/router_checkpoints/{size}/adapter"
    if not base or not adapter:
        typer.echo("need --size, or both --base and --adapter", err=True)
        raise typer.Exit(2)

    device, dtype = _device_dtype()
    typer.echo(f"[merge] base={base}\n[merge] adapter={adapter}\n[merge] device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(adapter)
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    except TypeError:  # transformers <4.49
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter)
    typer.echo("[merge] merging adapter into base weights…")
    merged = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    typer.echo(f"[merge] ✓ saved standalone merged model -> {out}")

    if push_to:
        from huggingface_hub import HfApi, create_repo
        create_repo(push_to, private=private, repo_type="model", exist_ok=True)
        HfApi().upload_folder(folder_path=str(out), repo_id=push_to, repo_type="model")
        typer.echo(f"[merge] ✓ pushed merged model -> https://huggingface.co/{push_to}")


if __name__ == "__main__":
    app()
