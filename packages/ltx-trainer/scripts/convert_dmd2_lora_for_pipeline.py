#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import typer
from safetensors.torch import load_file, save_file


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Convert DMD2 PEFT LoRA checkpoints to LTX pipeline LoRA key format.",
)


@app.command()
def main(
    input_path: str = typer.Argument(..., help="DMD2 student LoRA checkpoint saved by dmd2_trainer.py"),
    output_path: str = typer.Argument(..., help="Output LoRA checkpoint path for ltx_pipelines --lora"),
) -> None:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    state_dict = load_file(str(source))

    converted = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model."):
            key = key.removeprefix("base_model.model.")
        converted[key] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(destination))
    typer.echo(f"Converted {len(converted)} tensors: {source} -> {destination}")


if __name__ == "__main__":
    app()
