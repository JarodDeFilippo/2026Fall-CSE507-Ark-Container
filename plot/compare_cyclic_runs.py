#!/usr/bin/env python3
"""Compare student and teacher performance across multiple Ark+ runs."""

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from plot_cyclic_run import (
        INIT_POINT,
        MODELS,
        NATURE_PALETTE,
        SPLITS,
        _discover_run,
        _metric_label,
        _scale,
        _series,
        _slug,
        _task_name,
        _to_plot_units,
        xlim_from_first_point,
    )
    from run_labels import dataset_display, short_run_labels
except ImportError:  # pragma: no cover - supports ``python -m plot...``.
    from .plot_cyclic_run import (
        INIT_POINT,
        MODELS,
        NATURE_PALETTE,
        SPLITS,
        _discover_run,
        _metric_label,
        _scale,
        _series,
        _slug,
        _task_name,
        _to_plot_units,
        xlim_from_first_point,
    )
    from .run_labels import dataset_display, short_run_labels


EXPERIMENT_COLORS = NATURE_PALETTE


def _color_arg(text):
    """RUN=#RRGGBB, split on the last "=" so a run name may hold one."""
    name, _, color = text.rpartition("=")
    if not name or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        raise argparse.ArgumentTypeError(
            "{!r}: expected RUN=#RRGGBB".format(text)
        )
    return name, color


def _label_arg(text):
    """RUN=Label, split on the first "=" so a label may hold one."""
    name, _, label = text.partition("=")
    if not name or not label.strip():
        raise argparse.ArgumentTypeError(
            "{!r}: expected RUN=Label".format(text)
        )
    return name, label.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare student and teacher performance for shared datasets "
            "across two or more Ark+ runs."
        )
    )
    parser.add_argument(
        "run_directories",
        type=Path,
        nargs="+",
        help="Run directories containing evaluation/<dataset>/*.csv.",
    )
    parser.add_argument(
        "--metric",
        default="mAUC",
        help="Metric row to plot (default: mAUC).",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "both"),
        default="both",
        help=(
            "Evaluation split to plot. 'both' writes separate validation "
            "and test plots (default: both)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "comparisons",
        help="Parent output directory (default: plot/comparisons/).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Raster output resolution (default: 180).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help=(
            "Optional fixed lower y-axis limit, in percent for mAUC and "
            "the AUC_<class> metrics, in the metric's own units "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help=(
            "Optional fixed upper y-axis limit, in percent for mAUC and "
            "the AUC_<class> metrics, in the metric's own units "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--models",
        choices=("teacher", "student", "both"),
        default="both",
        help=(
            "Model series to draw. 'both' draws a dashed student and a "
            "solid teacher line per run; 'teacher' or 'student' draws only "
            "that series, solid, labelled with the run label alone "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--full-length",
        action="store_true",
        help=(
            "Draw every run to its own last evaluated cycle instead of "
            "truncating all runs at the shortest run's last cycle."
        ),
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        type=_color_arg,
        default=[],
        metavar="RUN=#RRGGBB",
        help=(
            "Fixed line colors keyed by run directory name, for example "
            "all4_cyclic_dinov3_sweep_S2_seed_100=#1f77b4. Runs not named "
            "keep the palette color of their position. Give it after the "
            "run directories, or end it with --."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=_label_arg,
        default=[],
        metavar="RUN=Label",
        help=(
            "Legend labels keyed by run directory name, for example "
            "all4_cyclic_dinov3_sweep_S2_seed_100=S2 (quote a label that "
            "holds spaces). Runs not named keep their short label; the "
            "output folder name does not change. Give it after the run "
            "directories, or end it with --."
        ),
    )
    args = parser.parse_args()
    for option, chosen in (
        ("--colors", args.colors),
        ("--labels", args.labels),
    ):
        names = [name for name, _ in chosen]
        if len(set(names)) != len(names):
            parser.error("{} names a run more than once".format(option))
    if len(args.run_directories) < 2:
        parser.error("at least two run directories are required")
    if (
        args.y_min is not None
        and args.y_max is not None
        and args.y_min >= args.y_max
    ):
        parser.error("--y-min must be less than --y-max")
    return args


def _y_limits(
        run_data_list,
        dataset,
        split,
        max_cycle,
        models,
        requested_min,
        requested_max,
        scale):
    """Axis limits in plot units; scale is full scale (100 for percent)."""
    values = []
    for run_data in run_data_list:
        for record in run_data.records[(dataset, split)]:
            if max_cycle is not None and record.cycle > max_cycle:
                continue
            for model in models:
                value = getattr(record, model)
                if math.isfinite(value):
                    values.append(value)

    if not values:
        return requested_min, requested_max

    data_min = min(values)
    data_max = max(values)
    span = max(data_max - data_min, 0.05 * scale)
    lower = (
        data_min - 0.08 * span
        if requested_min is None
        else requested_min
    )
    upper = (
        data_max + 0.08 * span
        if requested_max is None
        else requested_max
    )
    if data_min >= 0.0 and data_max <= scale:
        lower = max(0.0, lower)
        upper = min(scale, upper)
    return lower, upper


def _plot_line(
        ax,
        run_data,
        run_label,
        dataset,
        split,
        model,
        color,
        task_indices,
        task_count,
        max_cycle,
        single_model):
    records = [
        record
        for record in run_data.records[(dataset, split)]
        if max_cycle is None or record.cycle <= max_cycle
    ]
    x_values, y_values = _series(
        records,
        model,
        task_indices,
        task_count,
    )
    if not any(math.isfinite(value) for value in y_values):
        warnings.warn(
            "No {} values available for {} / {}".format(
                model,
                run_data.run_directory,
                dataset,
            )
        )
        return False

    # A lone series needs no student/teacher styling or legend suffix.
    solid = single_model or model == "teacher"
    # One finite point draws no line: show it as a marker (a fresh run's init point).
    lone = sum(math.isfinite(value) for value in y_values) == 1
    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle="-" if solid else "--",
        linewidth=2.0,
        alpha=0.99 if solid else 0.84,
        label=run_label if single_model else "{} — {}".format(
            run_label,
            model.title(),
        ),
        marker="o" if lone else None,
        markersize=6,
        clip_on=not lone,
    )
    return True


def _save_figure(fig, file_path, dpi):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset(
        run_data_list,
        run_labels,
        run_colors,
        dataset,
        split,
        output_directory,
        metric,
        dpi,
        output_format,
        requested_y_min,
        requested_y_max,
        models,
        full_length):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    plotted = 0
    completed_cycles = []

    for run_data in run_data_list:
        task_indices = {
            task: index
            for index, task in enumerate(run_data.task_order)
        }
        task_count = len(run_data.task_order)
        records = run_data.records[(dataset, split)]
        recognized_records = [
            record
            for record in records
            if record.evaluation_point == INIT_POINT
            or (
                record.evaluation_point.startswith("after_")
                and _task_name(record) in task_indices
            )
        ]
        if not recognized_records:
            raise ValueError(
                "No after-evaluation records found for {} in {}".format(
                    dataset,
                    split,
                )
            )
        completed_cycles.append(max(record.cycle for record in recognized_records))

    if full_length:
        # No cut: each run is drawn to its own last cycle.
        max_cycle = None
        last_cycle = max(completed_cycles)
    else:
        max_cycle = last_cycle = min(completed_cycles)

    for run_index, run_data in enumerate(run_data_list):
        task_indices = {
            task: index
            for index, task in enumerate(run_data.task_order)
        }
        task_count = len(run_data.task_order)
        color = run_colors[run_index]
        for model in models:
            plotted += _plot_line(
                ax,
                run_data,
                run_labels[run_index],
                dataset,
                split,
                model,
                color,
                task_indices,
                task_count,
                max_cycle,
                len(models) == 1,
            )

    y_limits = _y_limits(
        run_data_list,
        dataset,
        split,
        max_cycle,
        models,
        requested_y_min,
        requested_y_max,
        _scale(metric),
    )
    title = "{} — {} comparison".format(
        dataset_display(dataset),
        split.title(),
    )
    if not full_length:
        title += " through cycle {}".format(max_cycle)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(_metric_label(metric, split))
    xlim_from_first_point(ax, last_cycle + 0.05)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=12, integer=True, min_n_ticks=1)
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        # One series per run: a single column until there are more than 8 runs.
        columns = 1 if len(models) == 1 and len(run_data_list) <= 8 else 2
        ax.legend(frameon=False, ncol=columns)
    else:
        ax.text(
            0.5,
            0.5,
            "No model values are available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    file_path = output_directory / split / (
        "{}.{}".format(_slug(dataset), output_format)
    )
    _save_figure(fig, file_path, dpi)
    return file_path


def create_plots(args):
    run_data_list = [
        _to_plot_units(
            _discover_run(run_directory, args.metric),
            args.metric,
        )
        for run_directory in args.run_directories
    ]
    shared_datasets = [
        dataset
        for dataset in run_data_list[0].datasets
        if all(dataset in run_data.datasets for run_data in run_data_list[1:])
    ]
    if not shared_datasets:
        raise ValueError("The supplied runs have no shared datasets.")

    run_labels = short_run_labels(
        [run_data.run_directory for run_data in run_data_list]
    )
    run_name = "__vs__".join(
        _slug(label)
        for label in run_labels
    )
    output_directory = (
        args.output_root.expanduser().resolve() / run_name
    )
    splits = SPLITS if args.split == "both" else (args.split,)
    models = MODELS if args.models == "both" else (args.models,)
    overrides = dict(args.colors)
    run_colors = [
        overrides.get(
            run_data.run_directory.name,
            EXPERIMENT_COLORS[index % len(EXPERIMENT_COLORS)],
        )
        for index, run_data in enumerate(run_data_list)
    ]
    label_overrides = dict(args.labels)
    legend_labels = [
        label_overrides.get(run_data.run_directory.name, label)
        for label, run_data in zip(run_labels, run_data_list)
    ]
    output_paths = []
    for dataset in shared_datasets:
        for split in splits:
            output_paths.append(
                _plot_dataset(
                    run_data_list,
                    legend_labels,
                    run_colors,
                    dataset,
                    split,
                    output_directory,
                    args.metric,
                    args.dpi,
                    args.format,
                    args.y_min,
                    args.y_max,
                    models,
                    args.full_length,
                )
            )

    print("Runs: {}".format(", ".join(
        "{}={}".format(label, run_data.run_directory.name)
        for label, run_data in zip(run_labels, run_data_list)
    )))
    print("Shared datasets: {}".format(", ".join(shared_datasets)))
    if args.colors:
        print("Colors: {}".format(", ".join(
            "{}={}".format(label, color)
            for label, color in zip(legend_labels, run_colors)
        )))
    if args.labels:
        print("Labels: {}".format(", ".join(
            "{}={}".format(label, run_data.run_directory.name)
            for label, run_data in zip(legend_labels, run_data_list)
        )))
    compared = {run_data.run_directory.name for run_data in run_data_list}
    for option, chosen in (
        ("--colors", overrides),
        ("--labels", label_overrides),
    ):
        unused = sorted(set(chosen) - compared)
        if unused:
            print("Ignored {} names (no such compared run): {}".format(
                option, ", ".join(unused)
            ))
    print("Metric: {}".format(args.metric))
    if args.full_length:
        print("Each run is drawn to its own last completed cycle.")
    else:
        print("Each plot is truncated at the minimum completed cycle across runs.")
    print("Wrote {} plots to {}".format(len(output_paths), output_directory))
    return output_paths


def main():
    create_plots(parse_args())


if __name__ == "__main__":
    main()
