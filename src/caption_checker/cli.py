from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import click

from caption_checker.detect import detect
from caption_checker.evaluation import (
    CORPORA,
    SYSTEMS,
    CorpusError,
    ReadThroughSystem,
    ScoreReport,
    run_eval,
)
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


def _priming_option(uses: str = "."):
    """``--priming-term``, shared by ``check`` and ``correct``; ``uses`` ends
    the help text with where else the terms go."""
    return click.option(
        "--priming-term",
        "priming_terms",
        multiple=True,
        metavar="TERM",
        help="A Priming term for this transcript (a speaker, product, company; "
        "repeatable). Joins the domain vocabulary for this run" + uses,
    )


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
@_priming_option()
def check(
    file: Path,
    vocab_path: Path | None,
    out_format: str,
    oov_zipf: float | None,
    output: Path | None,
    priming_terms: tuple[str, ...],
) -> None:
    """Flag likely ASR errors in FILE (SRT/VTT) and suggest corrections.

    Exit status is 1 when any flag is emitted, 0 when the transcript looks clean.
    """
    config = _detect_config(oov_zipf)
    cues = parse(file)
    vocab = load_vocab(vocab_path, terms=priming_terms, algo=config.phonetic_algo)
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
    help="Decision-cache location (--per-flag only). "
    "Default: ~/.cache/caption-checker/corrections.json.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Do not read or write the decision cache (--per-flag only).",
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
    help="Print flag count, batch count and approximate cost of the pass "
    "that would run, then exit without calling the model.",
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
    "--oov-zipf",
    type=float,
    default=None,
    help="Override the OOV Zipf ceiling (passed through to detection).",
)
@_priming_option(", and is given to the Read-through directly.")
@click.option(
    "--per-flag",
    is_flag=True,
    default=False,
    help="Run the per-flag correction pass instead of the Read-through: the "
    "model sees only the flagged spans, after the internal-match bypass and "
    "the decision cache (the Read-through-off mode of ADR 0006).",
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
    oov_zipf: float | None,
    priming_terms: tuple[str, ...],
    per_flag: bool,
) -> None:
    """Detect likely ASR errors in FILE, judge corrections with an LLM, and
    write a corrected SRT/VTT plus a sidecar record of every flag.

    By default the LLM pass is the Read-through: the model reads the whole
    transcript in chunks, judges every flag and reports errors no detector
    raised (ADR 0006).

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
    from caption_checker import readthrough
    from caption_checker.corrector import MissingAPIKeyError, build_corrector

    if output is None:
        raise click.UsageError("pass -o PATH: correct needs an explicit output path")

    config = _detect_config(oov_zipf)
    cues = parse(file)
    vocab = load_vocab(vocab_path, terms=priming_terms, algo=config.phonetic_algo)
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

    read_through = not per_flag
    try:
        live = not estimate
        result = run_correction(
            cues,
            reviewer=reviewer,
            corrector=build_corrector(model) if live and not read_through else None,
            vocab=vocab,
            config=config,
            model_id=model,
            cache=cache,
            max_calls=max_calls,
            estimate_only=estimate,
            read_through=read_through,
            reader=readthrough.build_reader(model) if live and read_through else None,
            priming_terms=priming_terms,
        )
    except MissingAPIKeyError as exc:
        raise click.ClickException(str(exc)) from exc
    except MaxCallsExceededError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.estimate is not None:
        _print_estimate(result.estimate, model, read_through=read_through)
        return

    out_format = file.suffix.lower().lstrip(".")
    sidecar = sidecar_path(output)
    output.write_text(serialize(result.cues, format=out_format), encoding="utf-8")
    write_sidecar(sidecar, result.outcomes)
    if eval_out is not None:
        write_eval_table(eval_out, result.outcomes)

    _print_run_report(result, output, sidecar)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where Session/Transcript state persists "
    "(default: $CAPTION_CHECKER_DATA_DIR or ~/.local/share/caption-checker/web).",
)
def serve(host: str, port: int, data_dir: Path | None) -> None:
    """Start the locally-hosted web review UI."""
    # Imported here, not at module scope, so `check`/`correct` never pull in
    # the web stack (FastAPI/uvicorn/Jinja2) -- same rationale as `correct`'s
    # own lazy imports above.
    import uvicorn

    from caption_checker.web.app import create_app
    from caption_checker.web.storage import Storage, default_data_dir

    storage = Storage(data_dir or default_data_dir())
    app = create_app(storage)
    uvicorn.run(app, host=host, port=port)


@main.command("eval")
@click.option(
    "--corpus",
    "corpus_name",
    type=click.Choice(list(CORPORA)),
    default="scored",
    show_default=True,
    help="Named corpus to score. earnings21-* need `build-earnings21` first; "
    "earnings21-heldout (ADR 0006) and earnings21-heldout-2 (#28) are for a "
    "final comparison only.",
)
@click.option(
    "--system",
    "system_name",
    type=click.Choice(list(SYSTEMS)),
    default="local",
    show_default=True,
    help="System under test.",
)
@click.option(
    "--priming/--no-priming",
    default=False,
    show_default=True,
    help="Hand each transcript's Priming terms (an Earnings-21 call's company "
    "name) to the system under test.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenRouter model slug, for a system that calls one (read-through).",
)
@click.option(
    "--final",
    is_flag=True,
    default=False,
    help="Required to score a Held-out set: only for a final comparison, "
    "never while tuning (ADR 0006).",
)
def eval_(
    corpus_name: str, system_name: str, priming: bool, model: str, final: bool
) -> None:
    """Score a system under test on a named corpus and print recall (overall
    and by kind), case precision, Flag-level precision and cold-flag rate as
    separate numbers. Unlike the pytest Regression gate this has no floors."""
    from caption_checker.corrector import MissingAPIKeyError

    corpus = CORPORA[corpus_name]
    if corpus.held_out and not final:
        raise click.ClickException(
            f"{corpus_name} is a Held-out set, scored only for a final "
            "comparison (ADR 0006); pass --final if that is what this run is. "
            "Looking at it to motivate a change moves it to the Dev set."
        )
    try:
        system = SYSTEMS[system_name](model)
        report = run_eval(corpus, system, priming=priming)
    except (CorpusError, MissingAPIKeyError) as e:
        raise click.ClickException(str(e)) from e
    lines = [
        f"corpus: {corpus_name}",
        f"system: {system_name}"
        + (f" ({model})" if isinstance(system, ReadThroughSystem) else ""),
        f"priming: {'on' if priming else 'off'}",
        _render_score(report, corpus.headline_kinds),
    ]
    if isinstance(system, ReadThroughSystem):
        lines.append(_render_spend(system, report.audio_seconds))
    if corpus.caveat:
        lines.append(f"note: {corpus.caveat}")
    click.echo("\n".join(lines))


@main.command("build-earnings21")
def build_earnings21() -> None:
    """Download Earnings-21 (Google ASR output + Rev references, CC BY-SA 4.0)
    into the gitignored cache and build its Auto-labelled corpora: the
    eval-10 Held-out set, a Dev set of 5 other calls and heldout-2, a seeded
    draw of 10 of the rest. Raw files are fetched once; the corpora are
    rebuilt from them every run."""
    from caption_checker import earnings21

    summary = earnings21.build()
    for split, counts in summary.items():
        detail = ", ".join(
            f"{k} {v}" for k, v in sorted(counts.items()) if k not in ("calls", "cases")
        )
        click.echo(f"{split}: {counts['calls']} calls, {counts['cases']} cases ({detail})")
    click.echo(f"dropped (not cases): {', '.join(earnings21.DROPPED)}")
    click.echo(f"note: {earnings21.CAVEAT}")


def _render_score(report: ScoreReport, headline_kinds: tuple[str, ...] = ()) -> str:
    caught = report.true_positives
    should_flag = caught + report.false_negatives
    # With headline kinds, the overall number blends in kinds kept apart.
    label = "all-kinds recall" if headline_kinds else "recall"
    lines = [f"{label}: {report.recall:.3f} ({caught}/{should_flag})"]
    if headline_kinds:
        hit = sum(report.recall_by_kind.get(k, (0, 0))[0] for k in headline_kinds)
        total = sum(report.recall_by_kind.get(k, (0, 0))[1] for k in headline_kinds)
        rate = f"{hit / total:.3f}" if total else "n/a"
        lines.append(
            f"headline recall ({', '.join(headline_kinds)}): {rate} ({hit}/{total})"
        )
    for kind, (hit, total) in report.recall_by_kind.items():
        proposed, _ = report.with_candidate_by_kind.get(kind, (0, total))
        lines.append(
            f"  {kind}: {hit / total:.3f} ({hit}/{total}; "
            f"{proposed}/{total} with candidate)"
        )
    if report.entity_recall is not None:
        hit, total = report.entity_recall
        lines.append(f"  entity: {hit / total:.3f} ({hit}/{total})")
    should_not_flag = report.true_negatives + report.false_positives
    lines.append(
        f"case precision: {report.precision:.3f} "
        f"({report.true_negatives}/{should_not_flag})"
        if should_not_flag
        else "case precision: n/a (no should-not-flag cases)"
    )
    lines.append(
        f"flag-level precision: {report.flag_precision:.3f} "
        f"({report.flags_touching_errors}/{report.exhaustive_flags} flags "
        "on exhaustive sources)"
        if report.flag_precision is not None
        else "flag-level precision: n/a (no flags on exhaustive sources)"
    )
    lines.append(
        f"cold-flag rate: {report.cold_flag_rate:.3f} "
        f"({report.cold_flags}/{report.total_flags} flags)"
    )
    return "\n".join(lines)


def _render_spend(system: ReadThroughSystem, audio_seconds: float) -> str:
    hours = audio_seconds / 3600
    spend = system.spend
    if spend.cost_usd is None:
        cost = "cost: n/a (no cost reported)"
    else:
        per_hour = f"${spend.cost_usd / hours:.2f}" if hours else "n/a"
        cost = (
            f"cost: ${spend.cost_usd:.4f} for {hours:.3f} audio hours "
            f"({per_hour} per audio hour)"
        )
    cost += (
        f"; {spend.requests} requests, {spend.prompt_tokens} prompt + "
        f"{spend.completion_tokens} completion tokens"
    )
    return (
        f"{cost}\nread-through: {system.failed_chunks} failed chunks"
    )


def _detect_config(oov_zipf: float | None) -> DetectConfig:
    """The detection config both `check` and `correct` build from their shared
    pass-through options."""
    config = DetectConfig()
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


def _print_estimate(est: "Estimate", model: str, *, read_through: bool) -> None:
    cost = (
        f"${est.approx_cost_usd:.6f}"
        if est.approx_cost_usd is not None
        else f"n/a (~{est.approx_tokens} prompt tokens; no price on file for "
        "this model)"
    )
    lines = [
        f"pass: {'read-through' if read_through else 'per-flag'}",
        f"model: {model}",
        f"flags: {est.flag_count}",
    ]
    if not read_through:  # the Read-through has no bypass or cache
        lines.append(f"residue (after bypass + cache): {est.residue_count}")
    lines += [f"batches: {est.chunk_count}", f"approx cost: {cost}"]
    click.echo("\n".join(lines), err=True)


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
