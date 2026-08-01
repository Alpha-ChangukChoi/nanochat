"""
Compare two experiments by compute multiplier (CM).

For a target quality, the compute multiplier is the ratio

    CM = baseline EFLOPs to reach the target / variant EFLOPs to reach the target

so CM > 1 means the variant reaches the same quality with less compute (a win) and
CM < 1 means it needs more (a regression).

Only fully-annealed endpoints participate. Each experiment's ladder traces a
*frontier*: the final (EFLOPs, quality) point of every completed model, and CMs
are read off by inverting the baseline's frontier at each variant endpoint.
Mid-training curve points are never compared against: a mid-schedule model has
not had its LR anneal yet, so its loss sits above what a run *scheduled* to end
there would achieve, which systematically flatters the variant. (This bias bites
exactly when the experiments derive different horizons, e.g. an ablation that
changes the parameter count and thereby the token budget.)

Conventions:
- The anchor is each variant endpoint's quality: CM answers "how much compute does
  the baseline need to match what this variant model achieved".
- Frontier inversion is piecewise-linear in (log EFLOPs, metric). No parametric
  form (e.g. a power law) is assumed.
- Matched depths are not required: an anchor only needs to land within the
  baseline's frontier range. If it falls outside, the boundary segment is
  extrapolated linearly and the reported CM is marked with a trailing "?": a
  low-confidence result, but a stated one.

Three metrics are compared, each where it applies:
- val_bpb: the precise instrument (~±0.5% of compute), but only meaningful when
  both experiments trained on the same dataset.
- CORE: dataset-independent, so it also referees data ablations, but noisy
  (~±25% of compute per rung); a least-squares fit through each frontier gives
  an additional smoothed cm_fit per row.
- ChatCORE: the sft stage's aggregate score, same treatment as CORE; noisier
  still, so read it mostly through its cm_fit.

Usage:

    python -m scripts.compare <baseline_experiment> <variant_experiment>

Prints one `cm` record per variant model and metric (the log grammar, see
nanochat/logfmt.py). The records are also written to compare_vs_<baseline>.log in
the variant's experiment directory and, unless --plot=0, one png per stage is
saved next to it: compare_vs_<baseline>.png (2x2, one row per pretraining metric:
frontiers + CM), _sft.png (the same treatment for ChatCORE), and _inference.png
(the two latency <-> throughput sweeps overlaid).
"""

import os
import math
import argparse

from nanochat.common import get_experiment_dir
from nanochat.logfmt import format_record
from nanochat.experiment import (
    list_model_tags, read_base_summary, read_stage_summary, read_training_curve,
    read_bench_sweep, read_meta,
)


def list_completed_tags(experiment_dir):
    """Model tags whose pretraining finished (a summary record exists), sorted."""
    tags = list_model_tags(experiment_dir)
    completed = [tag for tag in tags if read_base_summary(experiment_dir, tag) is not None]
    return completed

# -----------------------------------------------------------------------------
# The compute multiplier calculation. One machinery for both metrics; the only
# difference between them is the direction of "better" (bpb decreases with
# compute, CORE increases), which the inversion handles order-agnostically.
# CORE is noisy (~±0.003 at d16+, which the ~0.08/decade slope turns into ~±10%
# of compute), so alongside the piecewise CM its caller also requests cm_fit:
# both frontiers are least-squares fit as y = a + b*log(eflops) and the fits are
# inverted instead, smoothing the noise at the cost of assuming a shared shape.
# (val_bpb gets no cm_fit: it is precise, and its frontier is visibly curved in
# (log eflops, bpb) space, so a single global line would misfit it.)

def read_frontier(experiment_dir, metric):
    """The (eflops, metric, tag) endpoints of all completed models, sorted by
    eflops. Every point is a fully-annealed model: comparisons only ever invert
    this frontier, never mid-training curves (see the module docstring).
    Pretraining metrics (val_bpb, core) come from the base summary; chatcore
    comes from the sft summary. The x axis is always *pretraining* eflops: the
    sft stage's compute is negligible next to it."""
    points = []
    for tag in list_completed_tags(experiment_dir):
        summary = read_base_summary(experiment_dir, tag)
        if metric == "chatcore":
            sft_log = os.path.join(experiment_dir, tag, "sft.log")
            sft_summary = read_stage_summary(sft_log) or {}
            sft_summary["eflops"] = summary["eflops"]
            summary = sft_summary
        if metric in summary:
            points.append((summary["eflops"], summary[metric], tag))
    points.sort()
    return points


def solve_segment(p0, p1, target):
    """The eflops where the straight line through p0,p1 in (log eflops, y) space
    reaches y=target. Works for interpolation and extrapolation alike."""
    (f0, y0), (f1, y1) = p0, p1
    w = (target - y0) / (y1 - y0)
    log_f = (1 - w) * math.log(f0) + w * math.log(f1)
    return math.exp(log_f)


def eflops_at_metric(points, target):
    """
    Invert a monotone (eflops, y) frontier: the eflops where it reaches y=target.
    Piecewise-linear in (log eflops, y); increasing (CORE) and decreasing (bpb)
    frontiers alike. If the target lies outside the measured range, the boundary
    segment is extrapolated. Returns (eflops, extrapolated) where extrapolated=True
    flags the low-confidence case, or None if the frontier is degenerate.
    """
    if len(points) < 2:
        return None
    # interpolation: find the first measured segment that crosses the target
    for p0, p1 in zip(points, points[1:]):
        (_, y0), (_, y1) = p0, p1
        if min(y0, y1) <= target <= max(y0, y1) and y0 != y1:
            eflops = solve_segment(p0, p1, target)
            return eflops, False
    # extrapolation: continue the boundary segment past the measured range
    increasing = points[-1][1] > points[0][1]
    before_start = (target < points[0][1]) == increasing # target precedes the smallest model
    segment = (points[0], points[1]) if before_start else (points[-2], points[-1])
    (_, y0), (_, y1) = segment
    if y0 == y1:
        return None # flat boundary segment: no slope to extrapolate along
    eflops = solve_segment(*segment, target)
    return eflops, True


def fit_frontier(points):
    """Least-squares fit y = a + b*log10(eflops) through a frontier. Returns (a, b)."""
    xs = [math.log10(f) for f, y in points]
    ys = [y for f, y in points]
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
    a = ybar - b * xbar
    return a, b


def invert_fit(fit, target):
    """The eflops where a fitted frontier y = a + b*log10(eflops) reaches y=target."""
    a, b = fit
    return 10 ** ((target - a) / b)


def compute_comparison(baseline_dir, variant_dir, metric, fit=False):
    """
    One row per variant endpoint with the metric: frontier-inverted CMs.
    Returns (rows, baseline_frontier, variant_frontier, baseline_fit, variant_fit),
    or None when either side has fewer than the 2 endpoints a frontier needs.
    """
    baseline_frontier = read_frontier(baseline_dir, metric)
    variant_frontier = read_frontier(variant_dir, metric)
    if len(baseline_frontier) < 2 or len(variant_frontier) < 2:
        print(f"{metric} comparison needs at least 2 completed models with {metric} on each side")
        return None
    baseline_points = [(f, y) for f, y, _ in baseline_frontier]
    variant_points = [(f, y) for f, y, _ in variant_frontier]
    baseline_fit = fit_frontier(baseline_points) if fit else None
    variant_fit = fit_frontier(variant_points) if fit else None
    rows = []
    for variant_eflops, anchor, tag in variant_frontier:
        baseline_result = eflops_at_metric(baseline_points, anchor)
        if baseline_result is None:
            print(f"{tag}: skipped, baseline frontier is degenerate at anchor {metric} {anchor:.6f}")
            continue
        baseline_eflops, extrapolated = baseline_result
        row = dict(
            model_tag=tag,
            anchor=anchor,
            baseline_eflops=baseline_eflops,
            variant_eflops=variant_eflops,
            cm=baseline_eflops / variant_eflops,
            extrapolated=extrapolated,
        )
        if fit:
            row["cm_fit"] = invert_fit(baseline_fit, anchor) / invert_fit(variant_fit, anchor)
        rows.append(row)
    return rows, baseline_frontier, variant_frontier, baseline_fit, variant_fit

# -----------------------------------------------------------------------------
# Plotting

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
BASELINE_COLOR = "#2a78d6"
VARIANT_COLOR = "#1baf7a"
EFLOP = 1e18 # records store EFLOPs; the plot axes are raw FLOPs

def draw_bpb_frontier(ax, baseline, variant, baseline_dir, variant_dir, bpb_result, comparable):
    """Panel: the two val_bpb endpoint frontiers. The training curves that produced
    each endpoint are drawn faded, for context only: they carry no weight in the
    comparison (their mid-schedule points are biased, see the module docstring)."""
    from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator

    _, baseline_frontier, variant_frontier, _, _ = bpb_result
    seen_tags = {}
    styles = [
        (baseline_dir, baseline_frontier, BASELINE_COLOR, f"{baseline} (baseline)"),
        (variant_dir, variant_frontier, VARIANT_COLOR, f"{variant} (variant)"),
    ]
    for experiment_dir, frontier, color, label in styles:
        for _, _, tag in frontier:
            points = read_training_curve(experiment_dir, tag)
            xs = [p[0] * EFLOP for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color=color, lw=1.2, marker="o", ms=2.5, alpha=0.25)
        xs = [f * EFLOP for f, y, tag in frontier]
        ys = [y for f, y, tag in frontier]
        ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=6, label=label)
        for f, y, tag in frontier:
            seen_tags.setdefault(tag, (f * EFLOP, y))
    for tag, (x, y) in seen_tags.items():
        ax.annotate(tag, (x, y), xytext=(6, -2), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel("val bpb", color=TEXT_PRIMARY)
    ax.set_title("final val_bpb per depth (faded: training curves, context only)", color=TEXT_PRIMARY, pad=12)
    # bpb spans well under a decade: label every 0.1-ish step, hide minor labels
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    if not comparable:
        ax.set_title("val bpb per depth: DIFFERENT val sets, not comparable", color=TEXT_PRIMARY, pad=12)


def draw_cm(ax, baseline, variant, rows, metric_label):
    """Panel: compute multipliers per rung vs variant training FLOPs. Open marker
    = extrapolated beyond the baseline frontier; dashed = fit-smoothed (CORE)."""
    xs = [row["variant_eflops"] * EFLOP for row in rows]
    cms = [row["cm"] for row in rows]
    ax.axhline(1.0, color=TEXT_SECONDARY, lw=1.2, ls="--", alpha=0.7)
    ax.annotate("baseline (CM=1)", (xs[-1], 1.0), xytext=(0, 6), textcoords="offset points",
                fontsize=9, color=TEXT_SECONDARY, ha="right")
    ax.plot(xs, cms, color=VARIANT_COLOR, lw=2.2, zorder=2, label="piecewise")
    cm_fits = []
    if "cm_fit" in rows[0]:
        cm_fits = [row["cm_fit"] for row in rows]
        ax.plot(xs, cm_fits, color=VARIANT_COLOR, lw=1.2, ls="--", alpha=0.6, zorder=2, label="fit-smoothed")
        ax.legend(frameon=False, loc="best", fontsize=9)
    for x, row in zip(xs, rows):
        facecolor = SURFACE if row["extrapolated"] else VARIANT_COLOR # open marker = extrapolated
        ax.plot([x], [row["cm"]], marker="o", ms=7, color=VARIANT_COLOR,
                markerfacecolor=facecolor, zorder=3)
        tag_label = f"{row['model_tag']}?" if row["extrapolated"] else row["model_tag"]
        ax.annotate(tag_label, (x, row["cm"]), xytext=(6, -10), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel(f"compute multiplier ({metric_label})", color=TEXT_PRIMARY)
    ax.set_title(f"CM of {variant} vs {baseline}, by {metric_label} frontier", color=TEXT_PRIMARY, pad=12)
    lo = min(cms + cm_fits + [1.0])
    hi = max(cms + cm_fits + [1.0])
    pad = 0.25 * (hi - lo) + 0.005
    ax.set_ylim(lo - pad, hi + pad)


def style_axes(fig, axes):
    """Shared cosmetics of the comparison figures."""
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", alpha=0.28, lw=0.6)
        ax.grid(True, which="minor", alpha=0.12, lw=0.5)
        ax.tick_params(colors=TEXT_SECONDARY)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(TEXT_SECONDARY)


def draw_frontier(ax, baseline, variant, result, metric_label):
    """Panel: the two ladder frontiers of one metric, points + least-squares fit lines."""
    rows, baseline_frontier, variant_frontier, baseline_fit, variant_fit = result
    styles = [
        (baseline_frontier, baseline_fit, BASELINE_COLOR, f"{baseline} (baseline)"),
        (variant_frontier, variant_fit, VARIANT_COLOR, f"{variant} (variant)"),
    ]
    for frontier, fit, color, label in styles:
        xs = [f * EFLOP for f, c, tag in frontier]
        ys = [c for f, c, tag in frontier]
        ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=6, label=label)
        fit_xs = [min(xs), max(xs)]
        fit_ys = [fit[0] + fit[1] * math.log10(x / EFLOP) for x in fit_xs]
        ax.plot(fit_xs, fit_ys, color=color, lw=1.2, ls="--", alpha=0.6)
        for f, c, tag in frontier:
            ax.annotate(tag, (f * EFLOP, c), xytext=(6, -2), textcoords="offset points",
                        fontsize=9, color=TEXT_SECONDARY, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("pretraining compute (FLOPs)", color=TEXT_PRIMARY)
    ax.set_ylabel(f"{metric_label} score", color=TEXT_PRIMARY)
    ax.set_title(f"final {metric_label} per depth (dashed: least-squares fits)", color=TEXT_PRIMARY, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)


def draw_note(ax, text):
    """Fill a panel with a note instead of data."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=TEXT_SECONDARY)
    ax.set_axis_off()


def make_plot(baseline, variant, baseline_dir, variant_dir, comparable, bpb_result, core_result, out_path):
    """One 2x2 figure: the val_bpb comparison on the top row, CORE on the bottom row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=150)
    (ax_bpb_frontier, ax_bpb_cm), (ax_core_frontier, ax_core_cm) = axes

    if bpb_result is not None:
        draw_bpb_frontier(ax_bpb_frontier, baseline, variant, baseline_dir, variant_dir, bpb_result, comparable)
    else:
        draw_note(ax_bpb_frontier, "val_bpb frontier needs 2+ completed models per side")
    if bpb_result is not None and comparable and bpb_result[0]:
        draw_cm(ax_bpb_cm, baseline, variant, bpb_result[0], "val_bpb")
    elif not comparable:
        draw_note(ax_bpb_cm, "val_bpb CM is undefined:\nthe experiments trained on different datasets")
    else:
        draw_note(ax_bpb_cm, "no comparable val_bpb points")

    if core_result is not None and core_result[0]:
        draw_frontier(ax_core_frontier, baseline, variant, core_result, "CORE")
        draw_cm(ax_core_cm, baseline, variant, core_result[0], "CORE")
    else:
        draw_note(ax_core_frontier, "CORE comparison needs 2+ scored models per side")
        draw_note(ax_core_cm, "CORE comparison needs 2+ scored models per side")

    live_axes = [ax for ax in axes.flat if ax.axison]
    style_axes(fig, live_axes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")


def make_sft_comparison_plot(baseline, variant, chatcore_result, out_path):
    """One 1x2 figure in the style of the CORE row: the two ChatCORE frontiers
    and the CM read off them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_frontier, ax_cm) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    draw_frontier(ax_frontier, baseline, variant, chatcore_result, "ChatCORE")
    draw_cm(ax_cm, baseline, variant, chatcore_result[0], "ChatCORE")
    style_axes(fig, [ax_frontier, ax_cm])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")


def make_inference_comparison_plot(baseline, variant, baseline_dir, variant_dir, out_path):
    """The two latency <-> throughput tradeoffs overlaid: one curve per model,
    colored by experiment, one point per batch size of its sweep. The serving
    comparison is deliberately visual (the "knee" of a curve resists a scalar):
    a variant curve above/right of its baseline peer is a strict serving win."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    # the two experiments' bs=1 points nearly coincide, so their model labels are
    # nudged apart vertically (baseline up, variant down) to stay readable
    styles = [
        (baseline_dir, BASELINE_COLOR, f"{baseline} (baseline)", 5),
        (variant_dir, VARIANT_COLOR, f"{variant} (variant)", -9),
    ]
    drew_any = False
    for experiment_dir, color, label, label_dy in styles:
        labeled = False # legend: one entry per experiment, not per curve
        for tag in list_completed_tags(experiment_dir):
            records = read_bench_sweep(experiment_dir, tag)
            if not records:
                continue
            xs = [r["tok_per_sec"] / r["batch"] for r in records]
            ys = [r["tok_per_sec"] for r in records]
            ax.plot(xs, ys, lw=2.0, marker="o", ms=4, color=color, label=None if labeled else label)
            labeled = True
            drew_any = True
            # the bs=1 point is the right end of the curve: name the model there
            ax.annotate(tag, (xs[0], ys[0]), xytext=(5, label_dy), textcoords="offset points",
                        fontsize=8, color=color)
    if not drew_any:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("tok/s per user (each batch element)", color=TEXT_PRIMARY)
    ax.set_ylabel("aggregate throughput (tok/s)", color=TEXT_PRIMARY)
    ax.set_title(f"throughput vs per-user speed: {variant} vs {baseline}", color=TEXT_PRIMARY, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(fig, [ax])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved plot to {out_path}")
    return True

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multipliers of a variant experiment over a baseline")
    parser.add_argument("baseline", type=str, help="Baseline experiment name")
    parser.add_argument("variant", type=str, help="Variant experiment name")
    parser.add_argument("--plot", type=int, default=1, help="save the comparison png (1) or not (0)")
    parser.add_argument("--plot-file", type=str, default=None, help="png path (default: <variant experiment dir>/compare_vs_<baseline>.png)")
    args = parser.parse_args()

    baseline_dir = get_experiment_dir(args.baseline)
    variant_dir = get_experiment_dir(args.variant)
    log_file = os.path.join(variant_dir, f"compare_vs_{args.baseline}.log")
    plot_file = args.plot_file
    if plot_file is None:
        plot_file = os.path.join(variant_dir, f"compare_vs_{args.baseline}.png")

    assert list_completed_tags(baseline_dir), f"No completed pretraining runs found in {baseline_dir}"
    assert list_completed_tags(variant_dir), f"No completed pretraining runs found in {variant_dir}"
    lines = []

    # the val_bpb comparison: only meaningful when both experiments trained on the same data
    # (the frontiers are still read regardless, so the plot can show them side by side)
    baseline_dataset = read_meta(baseline_dir).get("dataset")
    variant_dataset = read_meta(variant_dir).get("dataset")
    comparable = baseline_dataset == variant_dataset
    if not comparable:
        print(f"datasets differ ({baseline_dataset} vs {variant_dataset}): skipping the val_bpb comparison")
    bpb_result = compute_comparison(baseline_dir, variant_dir, "val_bpb")
    if comparable and bpb_result is not None:
        for row in bpb_result[0]:
            cm_str = f"{row['cm']:.4f}?" if row["extrapolated"] else f"{row['cm']:.4f}"
            record = format_record(
                "cm",
                metric="bpb",
                model_tag=row["model_tag"],
                anchor_bpb=round(row["anchor"], 6),
                baseline_eflops=round(row["baseline_eflops"], 4),
                variant_eflops=round(row["variant_eflops"], 4),
                cm=cm_str,
            )
            lines.append(record)

    # the CORE comparison: dataset-independent, works whenever both ladders have 2+ scores
    # the ChatCORE comparison: same treatment for the sft stage's aggregate score
    core_result = compute_comparison(baseline_dir, variant_dir, "core", fit=True)
    chatcore_result = compute_comparison(baseline_dir, variant_dir, "chatcore", fit=True)
    for metric, result in [("core", core_result), ("chatcore", chatcore_result)]:
        if result is None:
            continue
        for row in result[0]:
            cm_str = f"{row['cm']:.4f}?" if row["extrapolated"] else f"{row['cm']:.4f}"
            record = format_record(
                "cm",
                metric=metric,
                model_tag=row["model_tag"],
                **{f"anchor_{metric}": round(row["anchor"], 6)},
                baseline_eflops=round(row["baseline_eflops"], 4),
                variant_eflops=round(row["variant_eflops"], 4),
                cm=cm_str,
                cm_fit=round(row["cm_fit"], 4),
            )
            lines.append(record)

    # report to stdout and persist next to the experiment's other artifacts
    for line in lines:
        print(line)
    with open(log_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved records to {log_file}")

    if args.plot == 1:
        make_plot(args.baseline, args.variant, baseline_dir, variant_dir, comparable, bpb_result, core_result, plot_file)
        plot_base, plot_ext = os.path.splitext(plot_file)
        if chatcore_result is not None and chatcore_result[0]:
            make_sft_comparison_plot(args.baseline, args.variant, chatcore_result, f"{plot_base}_sft{plot_ext}")
        make_inference_comparison_plot(args.baseline, args.variant, baseline_dir, variant_dir, f"{plot_base}_inference{plot_ext}")
