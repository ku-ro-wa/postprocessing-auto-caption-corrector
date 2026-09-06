from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import click

from caption_checker.detect import detect
from caption_checker.models import DetectConfig, Flag
from caption_checker.parser import parse, serialize
from caption_checker.vocab import load_vocab


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


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--vocab",
    "vocab_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Extra domain terms (one per line) merged with the built-in list.",
)
@click.option(
    "--format",
    "out_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format. 'json' is the handoff shape for the LLM step.",
)
@click.option(
    "--no-embeddings",
    is_flag=True,
    default=False,
    help="Skip the local context-embedding detector.",
)
@click.option(
    "--oov-zipf",
    type=float,
    default=None,
    help="Override the OOV Zipf ceiling (default 0.0 = only true non-words; "
    "raise toward ~2.0 to also flag rare words).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the report here instead of stdout.",
)
def check(
    file: Path,
    vocab_path: Path | None,
    out_format: str,
    no_embeddings: bool,
    oov_zipf: float | None,
    output: Path | None,
) -> None:
    """Flag likely ASR errors in FILE (SRT/VTT) and suggest corrections.

    Exit status is 1 when any flag is emitted, 0 when the transcript looks clean.
    """
    config = DetectConfig(enable_embeddings=not no_embeddings)
    if oov_zipf is not None:
        config = replace(config, oov_zipf_max=oov_zipf)

    cues = parse(file)
    vocab = load_vocab(vocab_path, algo=config.phonetic_algo)
    flags = detect(cues, vocab=vocab, config=config)

    if out_format == "json":
        rendered = json.dumps([_flag_to_dict(f) for f in flags], indent=2)
    else:
        rendered = _render_text(flags, cues, file)

    if output is None:
        click.echo(rendered)
    else:
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"Wrote {len(flags)} flags to {output}", err=True)

    sys.exit(1 if flags else 0)


def _flag_to_dict(flag: Flag) -> dict:
    return {
        "span": flag.span,
        "global_indices": flag.global_indices,
        "cue_index": flag.cue_index,
        "start": flag.start.total_seconds(),
        "end": flag.end.total_seconds(),
        "detector": flag.detector,
        "reason": flag.reason,
        "candidates": flag.candidates,
        "confidence": flag.confidence,
        "context": flag.context,
    }


def _fmt_ts(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _render_text(flags: list[Flag], cues: list, file: Path) -> str:
    if not flags:
        return f"{file.name}: no likely caption errors found."

    by_cue: dict[int, list[Flag]] = {}
    for flag in flags:
        by_cue.setdefault(flag.cue_index, []).append(flag)

    cue_text = {c.index: c.text.replace("\n", " ") for c in cues}
    lines: list[str] = [
        f"{file.name}: {len(flags)} likely caption "
        f"{'error' if len(flags) == 1 else 'errors'}",
        "",
    ]
    for cue_index in sorted(by_cue):
        cue_flags = by_cue[cue_index]
        head = cue_flags[0]
        lines.append(
            f"[{_fmt_ts(head.start.total_seconds())} → "
            f"{_fmt_ts(head.end.total_seconds())}] cue {cue_index}"
        )
        text = cue_text.get(cue_index, "")
        for flag in cue_flags:
            text = text.replace(flag.span, f"»{flag.span}«", 1)
        lines.append(f"  {text}")
        for flag in cue_flags:
            suggestion = (
                f' → "{flag.candidates[0]}"' if flag.candidates else ""
            )
            extra = (
                f" (+{len(flag.candidates) - 1} more)"
                if len(flag.candidates) > 1
                else ""
            )
            lines.append(
                f"    » {flag.span}  [{flag.detector}]"
                f"{suggestion}{extra}  {flag.confidence:.2f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip()
