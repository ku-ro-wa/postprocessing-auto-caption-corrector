from __future__ import annotations

from pathlib import Path

import click

from caption_checker.parser import parse, serialize


@click.group()
def main() -> None:
    """Post-hoc ASR caption error checker."""


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the re-serialized output here instead of stdout.",
)
def roundtrip(file: Path, output: Path | None) -> None:
    """Parse FILE and re-serialize it, to verify the parser preserves content."""
    format = file.suffix.lower().lstrip(".")
    cues = parse(file)
    result = serialize(cues, format=format)
    if output is None:
        click.echo(result)
    else:
        output.write_text(result, encoding="utf-8")
        click.echo(f"Wrote {len(cues)} cues to {output}")
