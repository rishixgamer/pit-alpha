"""Packaged command line for synthetic and user-supplied pit-alpha data."""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml

from quantcheck.pit_alpha.config import load_config
from quantcheck.pit_alpha.example import write_example
from quantcheck.pit_alpha.reporting import write_reports
from quantcheck.pit_alpha.workflow import read_inputs, run

app = typer.Typer(help="Run timestamp-level QuantCheck factor research.", no_args_is_help=True)


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option(help="Path to the versioned YAML config.")],
) -> None:
    """Run the full workflow and publish JSON and Markdown together."""
    try:
        settings = load_config(config)
        observations, decisions, hashes = read_inputs(settings, config.parent)
        result = run(settings, observations, decisions)
        output = config.parent / settings.output_directory
        write_reports(output, result, hashes)
    except (ValueError, OSError, OverflowError, pl.exceptions.PolarsError, yaml.YAMLError) as error:
        typer.echo(f"pit-alpha: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Wrote {output / 'report.json'} and {output / 'report.md'}")


@app.command()
def example(
    directory: Annotated[Path, typer.Option(help="Directory for the synthetic fixture.")],
) -> None:
    """Generate the bundled, seeded Parquet example and YAML config."""
    try:
        config = write_example(directory)
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        typer.echo(f"pit-alpha: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Wrote {config}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
