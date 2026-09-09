from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import click

from caption_checker.detect import detect
from caption_checker.models import (
    DEFAULT_MODEL,
    DetectConfig,
    Flag,
    flag_to_dict,
    format_timestamp,
)
from caption_checker.parser import parse, serialize
from caption_checker.vocab import load_vocab

if TYPE_CHECKING:  # keeps the free `check` path from importing the LLM stack
    from caption_checker.correct import CorrectionResult, Estimate, Reviewer


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
    config = _detect_config(no_embeddings, oov_zipf)
    cues = parse(file)
    vocab = load_vocab(vocab_path, algo=config.phonetic_algo)
    flags = detect(cues, vocab=vocab, config=config)

    if out_format == "json":
        rendered = json.dumps([flag_to_dict(f) for f in flags], indent=2)
    else:
        rendered = _render_text(flags, cues, file)

    if output is None:
        click.echo(rendered)
    else:
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"Wrote {len(flags)} flags to {output}", err=True)

    sys.exit(1 if flags else 0)


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the corrected transcript. Required.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenRouter model slug for the correction pass.",
)
@click.option(
    "--yes-above",
    type=float,
    default=None,
    help="Non-interactive: apply every correction at or above this confidence, "
    "skip the rest. Required when stdin is not a TTY.",
)
@click.option(
    "--cache-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Decision-cache location. Default: ~/.cache/caption-checker/corrections.json.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Do not read or write the decision cache.",
)
@click.option(
    "--eval-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write a markdown eval table (verdict column left blank).",
)
@click.option(
    "--estimate",
    is_flag=True,
    default=False,
    help="Print flag count, batch count and approximate cost, then exit "
    "without calling the model.",
)
@click.option(
    "--max-calls",
    type=int,
    default=None,
    help="Abort before dispatching if the run would need more than N requests.",
)
@click.option(
    "--vocab",
    "vocab_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Extra domain terms (one per line) merged with the built-in list.",
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
    help="Override the OOV Zipf ceiling (passed through to detection).",
)
def correct(
    file: Path,
    output: Path | None,
    model: str,
    yes_above: float | None,
    cache_file: Path | None,
    no_cache: bool,
    eval_out: Path | None,
    estimate: bool,
    max_calls: int | None,
    vocab_path: Path | None,
    no_embeddings: bool,
    oov_zipf: float | None,
) -> None:
    """Detect likely ASR errors in FILE, judge corrections with an LLM, and
    write a corrected SRT/VTT plus a sidecar record of every flag.

    Unlike 'check' this writes files, can be interactive, and spends money;
    exit status is 0 on completion even if some flags were left uncorrected.
    """
    # Imported here, not at module scope, so a plain `check` run never pulls in
    # the correction orchestrator or the LLM backend (ADR-0002).
    from caption_checker.cache import DecisionCache, default_cache_path
    from caption_checker.correct import (
        InteractiveReviewer,
        MaxCallsExceededError,
        ThresholdReviewer,
        run_correction,
        sidecar_path,
        write_eval_table,
        write_sidecar,
    )
    from caption_checker.corrector import MissingAPIKeyError, build_corrector

    if output is None:
        raise click.UsageError("pass -o PATH: correct needs an explicit output path")

    config = _detect_config(no_embeddings, oov_zipf)
    cues = parse(file)
    vocab = load_vocab(vocab_path, algo=config.phonetic_algo)
    cache = DecisionCache.load(
        None if no_cache else (cache_file or default_cache_path()),
        enabled=not no_cache,
    )

    if yes_above is None and not estimate and not sys.stdin.isatty():
        _refuse_non_interactive(cues, vocab, config)

    reviewer: Reviewer
    if yes_above is not None:
        reviewer = ThresholdReviewer(yes_above)
    else:
        reviewer = InteractiveReviewer()

    try:
        result = run_correction(
            cues,
            reviewer=reviewer,
            corrector=None if estimate else build_corrector(model),
            vocab=vocab,
            config=config,
            model_id=model,
            cache=cache,
            max_calls=max_calls,
            estimate_only=estimate,
        )
    except MissingAPIKeyError as exc:
        raise click.ClickException(str(exc)) from exc
    except MaxCallsExceededError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.estimate is not None:
        _print_estimate(result.estimate, model)
        return

    out_format = file.suffix.lower().lstrip(".")
    sidecar = sidecar_path(output)
    output.write_text(serialize(result.cues, format=out_format), encoding="utf-8")
    write_sidecar(sidecar, result.outcomes)
    if eval_out is not None:
        write_eval_table(eval_out, result.outcomes)

    _print_run_report(result, output, sidecar)


def _detect_config(no_embeddings: bool, oov_zipf: float | None) -> DetectConfig:
    """The detection config both `check` and `correct` build from their shared
    pass-through options."""
    config = DetectConfig(enable_embeddings=not no_embeddings)
    if oov_zipf is not None:
        config = replace(config, oov_zipf_max=oov_zipf)
    return config


def _refuse_non_interactive(cues, vocab, config) -> None:
    """Non-TTY and no ``--yes-above``: fail naming a flag rather than hang or
    silently change nothing. Runs detection only so the message is concrete."""
    flags = detect(cues, vocab=vocab, config=config)
    if not flags:
        return
    first = min(flags, key=lambda f: (f.cue_index, min(f.global_indices)))
    raise click.ClickException(
        f"stdin is not a TTY and --yes-above was not given; cannot review "
        f"{first.span!r} at {format_timestamp(first.start)} "
        f"(cue {first.cue_index}). Pass --yes-above CONF to run unattended."
    )


def _print_estimate(est: "Estimate", model: str) -> None:
    cost = (
        f"${est.approx_cost_usd:.6f}"
        if est.approx_cost_usd is not None
        else f"n/a (~{est.approx_tokens} prompt tokens; no price on file for "
        "this model)"
    )
    click.echo(
        "\n".join(
            [
                f"model: {model}",
                f"flags: {est.flag_count}",
                f"residue (after bypass + cache): {est.residue_count}",
                f"batches: {est.chunk_count}",
                f"approx cost: {cost}",
            ]
        ),
        err=True,
    )


def _print_run_report(
    result: "CorrectionResult", output: Path, sidecar: Path
) -> None:
    counts: dict[str, int] = {}
    for o in result.outcomes:
        counts[o.outcome] = counts.get(o.outcome, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no flags"
    click.echo(
        f"Wrote {output} ({summary}); sidecar {sidecar.name}; "
        f"{result.corrector_calls} LLM request(s)",
        err=True,
    )


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
            f"[{format_timestamp(head.start)} → "
            f"{format_timestamp(head.end)}] cue {cue_index}"
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
