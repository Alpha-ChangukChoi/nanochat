"""
Render an experiment into a single report: experiments/<name>/report.md.

The report is the human-facing summary of everything the experiment produced.
It opens with the experiment's identity (meta.json) and then has one section per
pipeline stage that actually ran -- Pretraining, Inference, SFT -- each with its
own results table and plots. Table values render in units meant for human eyes
(13.7B tokens, 28.3GiB, 7.0h); the raw machine-readable numbers remain in the
stage logs and curve.log (see nanochat/logfmt.py). Markdown so it renders on
GitHub and stays greppable.

With --reference <experiment>, the report also includes the comparison section:
compute multipliers of this experiment over the reference (see scripts/compare.py),
with the 2x2 comparison figure saved alongside.

Usage: python -m scripts.report [experiment] [--reference <experiment>]
       (experiment defaults to $NANOCHAT_EXPERIMENT)
"""

import os
import math
import argparse
from datetime import datetime

from nanochat.common import get_experiment_dir, get_experiment_name
from nanochat.experiment import (
    read_meta, read_training_curve, read_base_summary, read_stage_summary, read_bench_sweep,
)
from nanochat.logfmt import parse_records
from scripts.curve import build_row
from scripts.compare import (
    list_completed_tags, compute_comparison, make_plot,
    make_sft_comparison_plot, make_inference_comparison_plot,
    read_frontier, fit_frontier, style_axes, draw_note, EFLOP,
    TEXT_PRIMARY, TEXT_SECONDARY, BASELINE_COLOR, VARIANT_COLOR,
)

# assumed price of one 8xH100 node hour, for the ballpark cost figure
DOLLARS_PER_NODE_HOUR = 24.0

# -----------------------------------------------------------------------------
# Formatting: raw summary values -> units meant for human consumption

# columns that are large counts (parameters, tokens), rendered as 286M / 13.7B
COUNT_COLUMNS = {"num_params", "num_scaling_params", "params", "tokens_trained", "total_batch_size"}
# columns that duplicate information already in the table (step is the training
# iteration of the loaded checkpoint, depth is the model tag)
HIDDEN_COLUMNS = {"step", "depth"}


def fmt_count(n):
    """A large count -> 3 significant figures with a suffix: 13.7B, 286M, 524K."""
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.3g}{suffix}"
    return f"{n:.0f}"


def fmt_bytes(n):
    """A byte count -> 3 significant figures with a byte suffix: 793MB, 6.31GB."""
    for div, suffix in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if abs(n) >= div:
            return f"{n / div:.3g}{suffix}"
    return f"{n:.0f}B"


def fmt_duration(sec):
    """Seconds -> 42s / 5.3m / 7.0h, whichever reads best."""
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def fmt_float(v):
    """A generic float at ~4 significant digits, never scientific notation."""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.4g}"
    return f"{v:.4f}"


def fmt_value(name, value):
    """One table cell: dispatch on the column name to pick the human unit."""
    if isinstance(value, str):
        return value
    if name in COUNT_COLUMNS:
        return fmt_count(value)
    # "_time_sec" (not "_sec": tok_per_sec is a rate) marks wall-clock durations
    if name.endswith("time_sec"):
        return fmt_duration(value)
    if name.endswith("_mib"):
        gib = value / 1024
        return f"{gib:.1f}GiB"
    if "bytes" in name:
        return fmt_bytes(value)
    if isinstance(value, float):
        return fmt_float(value)
    if isinstance(value, int) and abs(value) >= 10000:
        return f"{value:,}"
    return str(value)


# human column headers for the report tables, keyed by the summary field name;
# fields not listed fall back to a generic cleanup below
HEADER_NAMES = {
    "model_dim": "dim",
    "num_params": "params",
    "num_scaling_params": "scaling params",
    "num_iterations": "iters",
    "total_batch_size": "batch size",
    "tokens_trained": "tokens",
    "eflops": "EFLOPs",
    "param_data_ratio": "param:data",
    "train_time_sec": "train time",
    "peak_vram_mib": "peak VRAM",
    "val_bpb": "val bpb",
    "min_val_bpb": "min val bpb",
    "core": "CORE",
    "chatcore": "ChatCORE",
    "chatcore_cat": "ChatCORE (cat)",
}


def display_name(name):
    """The column header of a summary field, made for human eyes. Unit suffixes are
    dropped because the rendered values carry their own units ('5.3m', '28.3GiB')."""
    if name in HEADER_NAMES:
        return HEADER_NAMES[name]
    if name.endswith("time_sec"):
        name = name.removesuffix("_sec")
    if name.endswith("_mib"):
        name = name.removesuffix("_mib")
    return name.replace("_", " ")


def md_table(columns, rows):
    """Rows of dicts -> a markdown table string, one column per key."""
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = [str(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cm_str(row):
    """The compute multiplier of a comparison row, '?'-marked when extrapolated."""
    if row["extrapolated"]:
        return f"{row['cm']:.4f}?"
    return f"{row['cm']:.4f}"


# -----------------------------------------------------------------------------
# Stage tables

def stage_columns(rows, stage):
    """All 'stage.field' keys present in the rows, in first-seen (summary) order."""
    prefix = stage + "."
    columns = []
    for row in rows:
        for key in row:
            if key.startswith(prefix) and key not in columns:
                columns.append(key)
    return columns


def stage_table(rows, columns):
    """One stage's results table. Noise columns are hidden, values render in human
    units, and any column whose rendered value is identical across all models is
    collapsed into a prose line above the table (it is a fact about the stage, not
    about any particular model). Returns (prose, table_markdown)."""
    stripped = [column.split(".", 1)[1] for column in columns]
    pairs = [(c, s) for c, s in zip(columns, stripped) if s not in HIDDEN_COLUMNS]
    # min_val_bpb duplicates the final val_bpb on any fully-annealed run; it only
    # earns a column when some run got worse after its best point
    def duplicates_val_bpb(column):
        val_column = column.replace("min_val_bpb", "val_bpb")
        return all(row.get(column) == row.get(val_column) for row in rows)
    pairs = [(c, s) for c, s in pairs if not (s == "min_val_bpb" and duplicates_val_bpb(c))]
    # split the columns into constant (-> prose) and variable (-> table)
    constants = []
    variable = []
    for column, name in pairs:
        values = [fmt_value(name, row[column]) for row in rows if column in row]
        is_constant = len(rows) > 1 and len(values) == len(rows) and len(set(values)) == 1
        if is_constant:
            constants.append(f"{display_name(name)}={values[0]}")
        else:
            variable.append((column, name))
    table_rows = []
    for row in rows:
        table_row = {"model": row["model_tag"]}
        for column, name in variable:
            if column in row:
                table_row[display_name(name)] = fmt_value(name, row[column])
        table_rows.append(table_row)
    table = md_table(["model"] + [display_name(name) for _, name in variable], table_rows)
    prose = ""
    if constants:
        prose = "Constant across models: " + ", ".join(constants) + "."
    return prose, table


# -----------------------------------------------------------------------------
# Plots, one figure per stage section

def series(rows, x_key, y_key):
    """The (x, y) points of two row columns, keeping only rows that have both."""
    points = [(row[x_key], row[y_key]) for row in rows if x_key in row and y_key in row]
    return points


def draw_series(ax, points, scale, **kwargs):
    """One line of (x, y) points, x scaled to raw units (e.g. EFLOPs -> FLOPs)."""
    xs = [p[0] * scale for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, lw=2.0, marker="o", ms=5, **kwargs)


def make_pretraining_plot(experiment_dir, tags, out_path):
    """Two panels: the experiment's val_bpb training curves and its CORE frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator

    fig, (ax_curves, ax_core) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # left panel: val_bpb training curves (faded) + final points (bold), log-log
    ax = ax_curves
    finals = []
    for tag in tags:
        points = read_training_curve(experiment_dir, tag)
        xs = [p[0] * EFLOP for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=BASELINE_COLOR, lw=1.2, marker="o", ms=2.5, alpha=0.25)
        summary = read_base_summary(experiment_dir, tag)
        if summary is not None and "val_bpb" in summary:
            finals.append((summary["eflops"] * EFLOP, summary["val_bpb"], tag))
    finals.sort()
    ax.plot([p[0] for p in finals], [p[1] for p in finals],
            color=BASELINE_COLOR, lw=2.2, marker="o", ms=6)
    for x, y, tag in finals:
        ax.annotate(tag, (x, y), xytext=(6, -2), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel("val bpb", color=TEXT_PRIMARY)
    ax.set_title("final val_bpb per depth (faded: training curves)", color=TEXT_PRIMARY, pad=12)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # right panel: the CORE frontier + its least-squares fit
    ax = ax_core
    frontier = read_frontier(experiment_dir, "core")
    if len(frontier) >= 2:
        xs = [f * EFLOP for f, c, tag in frontier]
        ys = [c for f, c, tag in frontier]
        ax.plot(xs, ys, color=VARIANT_COLOR, lw=2.2, marker="o", ms=6)
        fit = fit_frontier([(f, c) for f, c, _ in frontier])
        fit_ys = [fit[0] + fit[1] * math.log10(x / EFLOP) for x in [min(xs), max(xs)]]
        ax.plot([min(xs), max(xs)], fit_ys, color=VARIANT_COLOR, lw=1.2, ls="--", alpha=0.6)
        for f, c, tag in frontier:
            ax.annotate(tag, (f * EFLOP, c), xytext=(6, -2), textcoords="offset points",
                        fontsize=9, color=TEXT_SECONDARY, va="center")
        ax.set_xscale("log")
    else:
        draw_note(ax, "CORE frontier needs 2+ scored models")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel("CORE score", color=TEXT_PRIMARY)
    ax.set_title("CORE frontier (dashed: least-squares fit)", color=TEXT_PRIMARY, pad=12)

    live_axes = [ax for ax in (ax_curves, ax_core) if ax.axison]
    style_axes(fig, live_axes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")


def make_inference_plot(sweeps, out_path):
    """The latency <-> throughput tradeoff. Each curve is one model; each point is
    one batch size of the sweep: batching amortizes the weight reads that dominate
    decode, buying aggregate throughput at the cost of per-request latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for i, (tag, records) in enumerate(sweeps.items()):
        xs = [r["tok_per_sec"] / r["batch"] for r in records]
        ys = [r["tok_per_sec"] for r in records]
        color = plt.cm.tab10(i % 10)
        ax.plot(xs, ys, lw=2.0, marker="o", ms=5, color=color, label=tag)
        # label every point with its batch size: each sweep ends at its own OOM point
        for record in records:
            x = record["tok_per_sec"] / record["batch"]
            y = record["tok_per_sec"]
            ax.annotate(f"({record['batch']})", (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.set_xlabel("tok/s per user (each batch element)", color=TEXT_PRIMARY)
    ax.set_ylabel("aggregate throughput (tok/s)", color=TEXT_PRIMARY)
    ax.set_title("throughput vs per-user speed (one point per batch size)", color=TEXT_PRIMARY, pad=12)
    ax.legend(frameon=False, fontsize=9)
    style_axes(fig, [ax])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")


def make_sft_plot(rows, out_path):
    """Two panels for the post-training story: per-task accuracies of the chat
    models against pretraining compute, and the aggregate scores (base CORE vs
    ChatCORE) on the same axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_tasks, ax_scores) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # left panel: per-task accuracies vs compute (chat evals, falling back to sft's)
    ax = ax_tasks
    stage = "chat." if any(any(k.startswith("chat.") for k in row) for row in rows) else "sft."
    # task columns are Capitalized (ARC-Easy, MMLU, ...), unlike the summary fields
    task_keys = []
    for row in rows:
        for key in row:
            if key.startswith(stage) and key[len(stage)].isupper() and key not in task_keys:
                task_keys.append(key)
    drawn = False
    for i, key in enumerate(task_keys):
        points = series(rows, "base.eflops", key)
        if not points:
            continue # a legend entry with no line helps nobody
        color = plt.cm.tab10(i % 10)
        draw_series(ax, points, EFLOP, color=color, label=key[len(stage):])
        drawn = True
    if drawn:
        ax.set_xscale("log")
        ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
        ax.set_ylabel("accuracy", color=TEXT_PRIMARY)
        ax.set_title(f"task evals vs compute ({stage.rstrip('.')} stage)", color=TEXT_PRIMARY, pad=12)
        ax.legend(frameon=False, fontsize=9)
    else:
        draw_note(ax, "no task evals yet")

    # right panel: the aggregate scores vs compute
    ax = ax_scores
    score_keys = [("base.core", "CORE (base)", BASELINE_COLOR), ("chat.chatcore", "ChatCORE (chat)", VARIANT_COLOR)]
    drawn = False
    for key, label, color in score_keys:
        points = series(rows, "base.eflops", key)
        if len(points) >= 1:
            draw_series(ax, points, EFLOP, color=color, label=label)
            drawn = True
    if drawn:
        ax.set_xscale("log")
        ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
        ax.set_ylabel("score", color=TEXT_PRIMARY)
        ax.set_title("aggregate scores vs compute", color=TEXT_PRIMARY, pad=12)
        ax.legend(frameon=False, fontsize=9)
    else:
        draw_note(ax, "no aggregate scores yet")

    live_axes = [ax for ax in (ax_tasks, ax_scores) if ax.axison]
    style_axes(fig, live_axes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")


# -----------------------------------------------------------------------------
# Report sections. Each stage section renders only if its stage actually ran,
# so a pretraining-only experiment gets a pretraining-only report.

def identity_section(name, experiment_dir, rows):
    """The report header: who this experiment is (meta.json) + the ladder totals."""
    meta = read_meta(experiment_dir)
    dirty_suffix = " (dirty tree, see code_diff.patch)" if meta.get("git_dirty") else ""
    lines = [f"# Experiment: {name}", ""]
    lines.append(f"- created: {meta.get('created', 'unknown')}")
    lines.append(f"- commit: `{meta.get('git_commit', 'unknown')[:12]}` on `{meta.get('git_branch', '?')}`{dirty_suffix}")
    lines.append(f"- dataset: `{meta.get('dataset', 'unknown')}`")
    lines.append(f"- report generated: {datetime.now().isoformat(timespec='seconds')}")
    # ladder totals: pretraining compute, tokens and wall-clock across the rungs
    total_eflops = sum(r.get("base.eflops", 0) for r in rows)
    if total_eflops > 0:
        total_tokens = sum(r.get("base.tokens_trained", 0) for r in rows)
        total_hours = sum(r.get("base.train_time_sec", 0) + r.get("sft.train_time_sec", 0) for r in rows) / 3600
        total_dollars = total_hours * DOLLARS_PER_NODE_HOUR
        lines.append(f"- ladder totals: {total_eflops:.1f} EFLOPs, {fmt_count(total_tokens)} tokens, "
                     f"{total_hours:.1f}h train wall-clock (~${total_dollars:.0f} at ${DOLLARS_PER_NODE_HOUR:.0f}/node-hour, excludes evals)")
    return "\n".join(lines)


def pretraining_section(rows, experiment_dir, tags, plot):
    """The base models: the results table + training curves and CORE frontier."""
    columns = stage_columns(rows, "base")
    if not columns:
        return None
    lines = ["## Pretraining", ""]
    prose, table = stage_table(rows, columns)
    if prose:
        lines.append(prose)
        lines.append("")
    lines.append(table)
    if plot:
        make_pretraining_plot(experiment_dir, tags, os.path.join(experiment_dir, "report_pretraining.png"))
        lines.append("")
        lines.append("![pretraining](report_pretraining.png)")
    return "\n".join(lines)


def inference_section(rows, experiment_dir, plot):
    """The inference bench of each base model: the headline serving metrics (TTFT,
    per-token decode latency, single-stream and batched throughput, MBU) and the
    latency <-> throughput tradeoff curve traced out by the batch size sweep."""
    sweeps = {}
    for row in rows:
        tag = row["model_tag"]
        records = read_bench_sweep(experiment_dir, tag)
        if records:
            sweeps[tag] = records
    if not sweeps:
        return None
    lines = ["## Inference", ""]
    # the bench setup, from any summary row (it is the same across models)
    setup = next((row for row in rows if "infer.gpu" in row), None)
    if setup is not None:
        batches = sorted({r["batch"] for records in sweeps.values() for r in records})
        batches_str = ", ".join(str(b) for b in batches)
        batches_min = batches[0]
        batches_max = batches[-1]
        lines.append(f"Benchmark on 1x {setup['infer.gpu']}: prefill a {setup['infer.prompt_tokens']}-token prompt "
                     f"(TTFT), then decode {setup['infer.decode_tokens']} tokens, sweeping the batch size "
                     f"{batches_min} -> {batches_max} (max bs = the largest batch that fit in VRAM). "
                     "tok/s/user is the decode speed each batch element sees. MBU and MFU are the achieved "
                     "fractions of peak GPU memory bandwidth and peak compute: decode is bandwidth-bound, "
                     "so MBU binds at small batch and MFU stays low until batching saturates compute.")
        lines.append("")
    columns = ["model", "params", "weights", "KV bytes/tok", "TTFT", "tok/s/user (bs=1)",
               "MBU (bs=1)", "max bs", "tok/s/user (max bs)", "tok/s (max bs)", "MFU (max bs)"]
    table_rows = []
    for row in rows:
        records = sweeps.get(row["model_tag"], [])
        by_batch = {r["batch"]: r for r in records}
        bs1 = by_batch.get(1)
        table_row = {"model": row["model_tag"]}
        if "infer.params" in row:
            table_row["params"] = fmt_count(row["infer.params"])
            table_row["weights"] = fmt_bytes(row["infer.weight_bytes"])
            table_row["KV bytes/tok"] = fmt_bytes(row["infer.kv_bytes_per_token"])
        if bs1 is not None:
            table_row["TTFT"] = f"{bs1['ttft_sec'] * 1e3:.1f}ms"
            table_row["tok/s/user (bs=1)"] = fmt_float(bs1["tok_per_sec"])
            table_row["MBU (bs=1)"] = f"{bs1['mbu_pct']:.1f}%"
        if records:
            # each model's sweep ends at its own capacity (the largest batch that fit)
            bs_max = records[-1]
            table_row["max bs"] = str(bs_max["batch"])
            table_row["tok/s/user (max bs)"] = fmt_float(bs_max["tok_per_sec"] / bs_max["batch"])
            table_row["tok/s (max bs)"] = fmt_float(bs_max["tok_per_sec"])
            table_row["MFU (max bs)"] = f"{bs_max['mfu_pct']:.2f}%"
        if len(table_row) > 1:
            table_rows.append(table_row)
    lines.append(md_table(columns, table_rows))
    if plot:
        make_inference_plot(sweeps, os.path.join(experiment_dir, "report_inference.png"))
        lines.append("")
        lines.append("![inference](report_inference.png)")
    return "\n".join(lines)


def sft_section(rows, experiment_dir, plot):
    """The chat models: SFT training columns merged with the chat eval columns
    (the chat eval is the official full evaluation of the SFT checkpoint; sft's
    own in-training task metrics are dropped in its favor when both exist)."""
    sft_columns = stage_columns(rows, "sft")
    chat_columns = stage_columns(rows, "chat")
    if not sft_columns and not chat_columns:
        return None
    chat_names = {column.split(".", 1)[1] for column in chat_columns}
    sft_columns = [column for column in sft_columns if column.split(".", 1)[1] not in chat_names]
    lines = ["## SFT", ""]
    if chat_columns:
        lines.append("Task accuracies and ChatCORE are the full chat eval of each SFT checkpoint.")
        lines.append("")
    prose, table = stage_table(rows, sft_columns + chat_columns)
    if prose:
        lines.append(prose)
        lines.append("")
    lines.append(table)
    if plot:
        make_sft_plot(rows, os.path.join(experiment_dir, "report_sft.png"))
        lines.append("")
        lines.append("![sft](report_sft.png)")
    return "\n".join(lines)


def ratio_cell(reference_value, experiment_value, fmt=fmt_float):
    """One side-by-side table cell: 'reference → this (ratio×)'."""
    ratio = experiment_value / reference_value
    return f"{fmt(reference_value)} → {fmt(experiment_value)} ({ratio:.2f}×)"


def inference_comparison_rows(reference_dir, experiment_dir):
    """Side-by-side serving metrics, one row per model tag benched in both
    experiments. Unlike the training CMs there is no quality-matching here: rows
    pair models of the same depth, which an ablation may make different sizes
    (the training comparison above referees what that difference is worth)."""
    rows = []
    for tag in list_completed_tags(experiment_dir):
        reference_summary = read_stage_summary(os.path.join(reference_dir, tag, "infer_bench.log"))
        experiment_summary = read_stage_summary(os.path.join(experiment_dir, tag, "infer_bench.log"))
        if reference_summary is None or experiment_summary is None:
            continue
        reference_sweep = read_bench_sweep(reference_dir, tag)
        experiment_sweep = read_bench_sweep(experiment_dir, tag)
        row = {"model": tag}
        row["params"] = ratio_cell(reference_summary["params"], experiment_summary["params"], fmt_count)
        row["KV bytes/token"] = ratio_cell(reference_summary["kv_bytes_per_token"], experiment_summary["kv_bytes_per_token"], fmt_bytes)
        row["max context rows"] = ratio_cell(reference_summary["max_full_context_rows"], experiment_summary["max_full_context_rows"], str)
        reference_bs1 = next((r["tok_per_sec"] for r in reference_sweep if r["batch"] == 1), None)
        experiment_bs1 = next((r["tok_per_sec"] for r in experiment_sweep if r["batch"] == 1), None)
        if reference_bs1 and experiment_bs1:
            row["tok/s bs=1"] = ratio_cell(reference_bs1, experiment_bs1)
        if reference_sweep and experiment_sweep:
            reference_peak = max(r["tok_per_sec"] for r in reference_sweep)
            experiment_peak = max(r["tok_per_sec"] for r in experiment_sweep)
            row["tok/s peak"] = ratio_cell(reference_peak, experiment_peak)
        rows.append(row)
    return rows


def comparison_section(reference, name, reference_dir, experiment_dir, plot_file, plot):
    """Compute multipliers of this experiment over the reference (see scripts/compare.py)."""
    lines = [f"## Comparison vs `{reference}`", ""]
    reference_dataset = read_meta(reference_dir).get("dataset")
    experiment_dataset = read_meta(experiment_dir).get("dataset")
    comparable = reference_dataset == experiment_dataset

    if not comparable:
        lines.append(f"datasets differ (`{reference_dataset}` vs `{experiment_dataset}`): "
                     "val_bpb is not comparable, CORE referees.")
        lines.append("")
    bpb_result = compute_comparison(reference_dir, experiment_dir, "val_bpb")
    if comparable and bpb_result is not None and bpb_result[0]:
        lines.append("Compute multipliers by val_bpb frontier (reference compute to match each of this "
                     "experiment's fully-annealed endpoints, over this experiment's compute; `?` = extrapolated):")
        lines.append("")
        table_rows = [dict(model=r["model_tag"], anchor_bpb=round(r["anchor"], 6),
                           reference_eflops=round(r["baseline_eflops"], 4),
                           eflops=round(r["variant_eflops"], 4), cm=cm_str(r)) for r in bpb_result[0]]
        lines.append(md_table(["model", "anchor_bpb", "reference_eflops", "eflops", "cm"], table_rows))
        lines.append("")

    core_result = compute_comparison(reference_dir, experiment_dir, "core", fit=True)
    chatcore_result = compute_comparison(reference_dir, experiment_dir, "chatcore", fit=True)
    score_tables = [
        ("core", core_result, "Compute multipliers by CORE frontier (noisy, ~±25% per rung; "
                              "cm_fit smooths via least-squares frontier fits):"),
        ("chatcore", chatcore_result, "Compute multipliers by ChatCORE frontier (the sft stage's "
                                      "aggregate score; noisier still than CORE):"),
    ]
    for metric, result, caption in score_tables:
        if result is None or not result[0]:
            continue
        lines.append(caption)
        lines.append("")
        anchor_column = f"anchor_{metric}"
        table_rows = [{"model": r["model_tag"], anchor_column: round(r["anchor"], 4),
                       "reference_eflops": round(r["baseline_eflops"], 4),
                       "eflops": round(r["variant_eflops"], 4), "cm": cm_str(r),
                       "cm_fit": round(r["cm_fit"], 4)} for r in result[0]]
        lines.append(md_table(["model", anchor_column, "reference_eflops", "eflops", "cm", "cm_fit"], table_rows))
        lines.append("")

    infer_rows = inference_comparison_rows(reference_dir, experiment_dir)
    if infer_rows:
        lines.append("Inference, side by side (`reference → this (ratio×)` at matched depth; note matched "
                     "depth is not matched quality -- the training comparison above referees quality):")
        lines.append("")
        columns = ["model", "params", "tok/s bs=1", "tok/s peak", "KV bytes/token", "max context rows"]
        lines.append(md_table(columns, infer_rows))
        lines.append("")

    if plot:
        make_plot(reference, name, reference_dir, experiment_dir, comparable, bpb_result, core_result, plot_file)
        lines.append(f"![comparison]({os.path.basename(plot_file)})")
        plot_base, plot_ext = os.path.splitext(plot_file)
        if chatcore_result is not None and chatcore_result[0]:
            sft_plot_file = f"{plot_base}_sft{plot_ext}"
            make_sft_comparison_plot(reference, name, chatcore_result, sft_plot_file)
            lines.append("")
            lines.append(f"![sft comparison]({os.path.basename(sft_plot_file)})")
        infer_plot_file = f"{plot_base}_inference{plot_ext}"
        drew_inference = make_inference_comparison_plot(reference, name, reference_dir, experiment_dir, infer_plot_file)
        if drew_inference:
            lines.append("")
            lines.append(f"![inference comparison]({os.path.basename(infer_plot_file)})")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an experiment's report.md")
    parser.add_argument("experiment", type=str, nargs="?", default=None, help="Experiment name (default: $NANOCHAT_EXPERIMENT)")
    parser.add_argument("--reference", type=str, default=None, help="Reference experiment to compute the comparison section against")
    parser.add_argument("--plot", type=int, default=1, help="render the pngs (1) or not (0)")
    args = parser.parse_args()

    name = args.experiment if args.experiment is not None else get_experiment_name()
    experiment_dir = get_experiment_dir(name)
    tags = list_completed_tags(experiment_dir)
    assert tags, f"No completed pretraining runs found in {experiment_dir}"

    rows = [build_row(os.path.join(experiment_dir, tag), tag) for tag in tags]
    plot = args.plot == 1
    sections = []
    sections.append(identity_section(name, experiment_dir, rows))
    for section in [
        pretraining_section(rows, experiment_dir, tags, plot),
        inference_section(rows, experiment_dir, plot),
        sft_section(rows, experiment_dir, plot),
    ]:
        if section is not None:
            sections.append(section)
    if args.reference is not None:
        reference_dir = get_experiment_dir(args.reference)
        compare_plot_file = os.path.join(experiment_dir, f"compare_vs_{args.reference}.png")
        sections.append(comparison_section(args.reference, name, reference_dir, experiment_dir, compare_plot_file, plot))

    report_path = os.path.join(experiment_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"saved report to {report_path}")
