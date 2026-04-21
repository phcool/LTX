#!/usr/bin/env python

from pathlib import Path

import typer
import yaml

from ltx_trainer.dmd2_config import Dmd2TrainerConfig
from ltx_trainer.dmd2_trainer import DMD2Trainer

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train LTX-2 with a DMD2-style video distillation objective.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    disable_progress_bars: bool = typer.Option(False, "--disable-progress-bars"),
) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as handle:
        config_data = yaml.safe_load(handle)

    try:
        config = Dmd2TrainerConfig(**config_data)
    except Exception as exc:
        typer.echo(f"Error: invalid configuration data: {exc}")
        raise typer.Exit(code=1) from exc

    DMD2Trainer(config).train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
