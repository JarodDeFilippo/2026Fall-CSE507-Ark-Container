#!/usr/bin/env python3
"""Compare student and teacher performance across multiple Ark+ runs."""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from plot_cyclic_run import (
        MODELS,
        NATURE_PALETTE,
        SPLITS,
        _discover_run,
        _metric_label,
        _series,
        _slug,
        _task_name,
    )
    from run_labels import short_run_labels
except ImportError:  # pragma: no cover - supports ``python -m plot...``.
    from .plot_cyclic_run import (
        MODELS,
        NATURE_PALETTE,
        SPLITS,
        _discover_run,
        _metric_label,
        _series,
        _slug,
        _task_name,
    )
    from .run_labels import short_run_labels


EXPERIMENT_COLORS = NATURE_PALETTE


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
        help="Optional fixed lower y-axis limit.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed upper y-axis limit.",
    )
    args = parser.parse_args()
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
        requested_min,
        requested_max):
    values = []
    for run_data in run_data_list:
        for record in run_data.records[(dataset, split)]:
            if record.cycle > max_cycle:
                continue
            for value in (record.student, record.teacher):
                if math.isfinite(value):
                    values.append(value)

    if not values:
        return requested_min, requested_max

    data_min = min(values)
    data_max = max(values)
    span = max(data_max - data_min, 0.05)
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
    if data_min >= 0.0 and data_max <= 1.0:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
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
        max_cycle):
    records = [
        record
        for record in run_data.records[(dataset, split)]
        if record.cycle <= max_cycle
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

    ax.plot(
        x_values,
        y_values,
        color=color,
        linestyle="-" if model == "teacher" else "--",
        linewidth=2.0,
        alpha=0.84 if model == "student" else 0.99,
        label="{} — {}".format(
            run_label,
            model.title(),
        ),
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
        dataset,
        split,
        output_directory,
        metric,
        dpi,
        output_format,
        requested_y_min,
        requested_y_max):
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
            if record.evaluation_point.startswith("after_")
            and _task_name(record) in task_indices
        ]
        if not recognized_records:
            raise ValueError(
                "No after-evaluation records found for {} in {}".format(
                    dataset,
                    split,
                )
            )
        completed_cycles.append(max(record.cycle for record in recognized_records))

    max_cycle = min(completed_cycles)

    for run_index, run_data in enumerate(run_data_list):
        task_indices = {
            task: index
            for index, task in enumerate(run_data.task_order)
        }
        task_count = len(run_data.task_order)
        color = EXPERIMENT_COLORS[run_index % len(EXPERIMENT_COLORS)]
        for model in MODELS:
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
            )

    y_limits = _y_limits(
        run_data_list,
        dataset,
        split,
        max_cycle,
        requested_y_min,
        requested_y_max,
    )
    ax.set_title(
        "{} — {} comparison through cycle {}".format(
            dataset,
            split.title(),
            max_cycle,
        ),
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel(_metric_label(metric))
    ax.set_xlim(0, max_cycle + 0.05)
    ax.set_ylim(*y_limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        ax.legend(frameon=False, ncol=2)
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
        _discover_run(run_directory, args.metric)
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
    output_paths = []
    for dataset in shared_datasets:
        for split in splits:
            output_paths.append(
                _plot_dataset(
                    run_data_list,
                    run_labels,
                    dataset,
                    split,
                    output_directory,
                    args.metric,
                    args.dpi,
                    args.format,
                    args.y_min,
                    args.y_max,
                )
            )

    print("Runs: {}".format(", ".join(
        "{}={}".format(label, run_data.run_directory.name)
        for label, run_data in zip(run_labels, run_data_list)
    )))
    print("Shared datasets: {}".format(", ".join(shared_datasets)))
    print("Metric: {}".format(args.metric))
    print("Each plot is truncated at the minimum completed cycle across runs.")
    print("Wrote {} plots to {}".format(len(output_paths), output_directory))
    return output_paths


def main():
    create_plots(parse_args())


if __name__ == "__main__":
    main()
