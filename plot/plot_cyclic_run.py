#!/usr/bin/env python3
"""Plot focused and unfocused performance from an Ark+ run."""

import argparse
import csv
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REQUIRED_COLUMNS = {
    "cycle",
    "epoch",
    "evaluation_point",
    "metric",
    "student",
    "teacher",
}
SPLITS = ("validation", "test")
MODELS = ("student", "teacher")
NATURE_PALETTE = (
    "#6C94AE",  # softened cyan-blue
    "#C67F6F",  # softened vermilion
    "#6D9F8D",  # softened green
    "#7585A7",  # softened indigo
    "#C7A270",  # softened ochre
    "#9AA6C0",  # softened slate
    "#8CBDB2",  # softened mint
    "#B66A68",  # softened red
    "#8E7662",  # softened brown
    "#A8947D",  # softened tan
)
MODEL_COLORS = {
    "student": NATURE_PALETTE[0],
    "teacher": NATURE_PALETTE[1],
}
UNFOCUSED_COLORS = {
    ("student", "validation"): "#A9C4D7",
    ("student", "test"): NATURE_PALETTE[0],
    ("teacher", "validation"): "#D9ACA1",
    ("teacher", "test"): NATURE_PALETTE[1],
}


@dataclass(frozen=True)
class EvaluationRecord:
    dataset: str
    split: str
    cycle: int
    epoch: int
    evaluation_point: str
    student: float
    teacher: float
    source_order: int


@dataclass(frozen=True)
class RunData:
    run_directory: Path
    datasets: tuple
    task_order: tuple
    records: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create focused, unfocused, and combined performance plots "
            "from an Ark+ training run."
        )
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run directory containing evaluation/<dataset>/*.csv.",
    )
    parser.add_argument(
        "--metric",
        default="mAUC",
        help="Metric row to plot (default: mAUC).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Parent output directory. Figures are written beneath "
            "<output-root>/<run-name> (default: plot/)."
        ),
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
    return parser.parse_args()


def _parse_number(value):
    value = value.strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _read_metric_rows(file_path, dataset, split, metric):
    records = []
    available_metrics = set()
    with file_path.open(newline="") as file_descriptor:
        reader = csv.DictReader(file_descriptor)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS.difference(columns)
        if missing_columns:
            raise ValueError(
                "{} is missing required columns: {}".format(
                    file_path,
                    ", ".join(sorted(missing_columns)),
                )
            )

        for source_order, row in enumerate(reader):
            row_metric = row["metric"].strip()
            available_metrics.add(row_metric)
            if row_metric != metric:
                continue
            evaluation_point = row["evaluation_point"].strip()
            if not evaluation_point:
                continue
            try:
                cycle = int(float(row["cycle"]))
                epoch = int(float(row["epoch"]))
            except ValueError as error:
                raise ValueError(
                    "Invalid cycle or epoch in {} at row {}".format(
                        file_path,
                        source_order + 2,
                    )
                ) from error
            records.append(
                EvaluationRecord(
                    dataset=dataset,
                    split=split,
                    cycle=cycle,
                    epoch=epoch,
                    evaluation_point=evaluation_point,
                    student=_parse_number(row["student"]),
                    teacher=_parse_number(row["teacher"]),
                    source_order=source_order,
                )
            )

    if not records:
        raise ValueError(
            "No '{}' rows with an evaluation point found in {}. "
            "Available metrics: {}".format(
                metric,
                file_path,
                ", ".join(sorted(available_metrics)),
            )
        )
    return records


def _infer_task_order(records):
    candidate_series = max(
        records.values(),
        key=lambda series: len(
            {record.evaluation_point for record in series}
        ),
    )
    earliest_cycle = min(record.cycle for record in candidate_series)
    task_order = []
    for record in candidate_series:
        if record.cycle != earliest_cycle:
            continue
        if not record.evaluation_point.startswith("after_"):
            continue
        dataset = record.evaluation_point[len("after_"):]
        if dataset and dataset not in task_order:
            task_order.append(dataset)
    if not task_order:
        raise ValueError(
            "Could not infer evaluation order. Expected evaluation points "
            "such as 'after_VinDrCXR' or 'after_joint'."
        )
    return tuple(task_order)


def _discover_run(run_directory, metric):
    run_directory = run_directory.expanduser().resolve()
    evaluation_directory = run_directory / "evaluation"
    if not evaluation_directory.is_dir():
        raise FileNotFoundError(
            "Evaluation directory does not exist: {}".format(
                evaluation_directory
            )
        )

    dataset_directories = sorted(
        path for path in evaluation_directory.iterdir() if path.is_dir()
    )
    if not dataset_directories:
        raise ValueError(
            "No dataset directories found under {}".format(
                evaluation_directory
            )
        )

    records = {}
    for dataset_directory in dataset_directories:
        dataset = dataset_directory.name
        for split, file_name in (
            ("validation", "val_performance.csv"),
            ("test", "test_performance.csv"),
        ):
            file_path = dataset_directory / file_name
            if not file_path.is_file():
                raise FileNotFoundError(
                    "Missing {} evaluation file for {}: {}".format(
                        split,
                        dataset,
                        file_path,
                    )
                )
            records[(dataset, split)] = _read_metric_rows(
                file_path,
                dataset,
                split,
                metric,
            )

    task_order = _infer_task_order(records)
    discovered = tuple(path.name for path in dataset_directories)
    datasets = tuple(
        dataset for dataset in task_order if dataset in discovered
    ) + tuple(
        dataset for dataset in discovered if dataset not in task_order
    )
    return RunData(
        run_directory=run_directory,
        datasets=datasets,
        task_order=task_order,
        records=records,
    )


def _task_name(record):
    return record.evaluation_point[len("after_"):]


def _ordered_records(records, task_indices):
    recognized = [
        record
        for record in records
        if record.evaluation_point.startswith("after_")
        and _task_name(record) in task_indices
    ]
    return sorted(
        recognized,
        key=lambda record: (
            record.cycle,
            task_indices[_task_name(record)],
            record.source_order,
        ),
    )


def _series(records, model, task_indices, task_count):
    ordered = _ordered_records(records, task_indices)
    x_values = [
        record.cycle - 1
        + (task_indices[_task_name(record)] + 1) / task_count
        for record in ordered
    ]
    y_values = [getattr(record, model) for record in ordered]
    return x_values, y_values


def _focused_records(records, dataset, evaluation_order):
    focus_point = "after_{}".format(dataset)
    focused = [
        record
        for record in records
        if record.evaluation_point == focus_point
    ]
    if focused:
        return focused

    # Joint/concurrent training evaluates after one shared event rather than
    # after a dataset-specific task. Every such point is focused for each
    # dataset because all datasets participate in the update. The inferred
    # event name is used instead of hard-coding only ``after_joint`` so other
    # concurrent exporters can use names such as ``after_concurrent``.
    if len(evaluation_order) == 1 and evaluation_order[0] != dataset:
        shared_point = "after_{}".format(evaluation_order[0])
        return [
            record
            for record in records
            if record.evaluation_point == shared_point
        ]
    return []


def _unfocused_records(records):
    return list(records)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _metric_label(metric):
    if metric == "mAUC":
        return "Mean AUROC"
    return metric.replace("_", " ").title()


def _y_limits(run_data, requested_min, requested_max):
    values = [
        value
        for records in run_data.records.values()
        for record in records
        for value in (record.student, record.teacher)
        if math.isfinite(value)
    ]
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


def _configure_axis(ax, title, metric, y_limits, max_cycle):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(_metric_label(metric))
    ax.set_ylim(*y_limits)
    ax.set_xlim(0, max_cycle + 0.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, file_path, dpi):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_model_line(
        ax,
        records,
        model,
        split,
        task_indices,
        task_count,
        label,
        linestyle,
        alpha=None,
        linewidth=2.0,
        color=None):
    x_values, y_values = _series(
        records,
        model,
        task_indices,
        task_count,
    )
    if not any(math.isfinite(value) for value in y_values):
        warnings.warn("No values available for {}".format(label))
        return False
    if alpha is None:
        alpha = 0.84 if model == "student" else 0.99
    ax.plot(
        x_values,
        y_values,
        color=color or MODEL_COLORS[model],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    return True


def _plot_unfocused_lines(
        ax,
        run_data,
        dataset,
        split,
        task_indices,
        task_count):
    plotted = 0
    records = _unfocused_records(
        run_data.records[(dataset, split)]
    )
    for model in MODELS:
        plotted += _plot_model_line(
            ax,
            records,
            model,
            split,
            task_indices,
            task_count,
            "Unfocused {}".format(model.title()),
            "--",
            alpha=0.44 if model == "student" else 0.56,
            linewidth=1.35,
            color=UNFOCUSED_COLORS[(model, split)],
        )
    return plotted


def _plot_focused(
        run_data,
        dataset,
        split,
        output_directory,
        args,
        y_limits,
        max_cycle):
    task_indices = {
        task: index for index, task in enumerate(run_data.task_order)
    }
    task_count = len(run_data.task_order)
    records = _focused_records(
        run_data.records[(dataset, split)],
        dataset,
        run_data.task_order,
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for model in MODELS:
        _plot_model_line(
            ax,
            records,
            model,
            split,
            task_indices,
            task_count,
            "{} {}".format(model.title(), split.title()),
            "-",
            linewidth=2.25,
        )
    _configure_axis(
        ax,
        "{} — Focused {} Performance".format(
            dataset,
            split.title(),
        ),
        args.metric,
        y_limits,
        max_cycle,
    )
    ax.legend(frameon=False, ncol=2)
    file_path = output_directory / "focused" / split / (
        "{}.{}".format(_slug(dataset), args.format)
    )
    _save_figure(fig, file_path, args.dpi)
    return file_path


def _plot_unfocused(
        run_data,
        dataset,
        split,
        output_directory,
        args,
        y_limits,
        max_cycle):
    task_indices = {
        task: index for index, task in enumerate(run_data.task_order)
    }
    task_count = len(run_data.task_order)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    plotted = _plot_unfocused_lines(
        ax,
        run_data,
        dataset,
        split,
        task_indices,
        task_count,
    )
    _configure_axis(
        ax,
        "{} — All {} Evaluation Performance".format(
            dataset,
            split.title(),
        ),
        args.metric,
        y_limits,
        max_cycle,
    )
    if plotted:
        ax.legend(frameon=False, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "No recognized evaluations are available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    file_path = output_directory / "unfocused" / split / (
        "{}.{}".format(_slug(dataset), args.format)
    )
    _save_figure(fig, file_path, args.dpi)
    return file_path


def _plot_focused_unfocused(
        run_data,
        dataset,
        split,
        output_directory,
        args,
        y_limits,
        max_cycle):
    task_indices = {
        task: index for index, task in enumerate(run_data.task_order)
    }
    task_count = len(run_data.task_order)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    _plot_unfocused_lines(
        ax,
        run_data,
        dataset,
        split,
        task_indices,
        task_count,
    )
    focused = _focused_records(
        run_data.records[(dataset, split)],
        dataset,
        run_data.task_order,
    )
    for model in MODELS:
        _plot_model_line(
            ax,
            focused,
            model,
            split,
            task_indices,
            task_count,
            "Focused {} {}".format(
                model.title(),
                split.title(),
            ),
            "-",
            linewidth=2.4,
        )
    _configure_axis(
        ax,
        "{} — Focused vs. Unfocused ({} Focus)".format(
            dataset,
            split.title(),
        ),
        args.metric,
        y_limits,
        max_cycle,
    )
    ax.legend(frameon=False, ncol=2)
    file_path = output_directory / "focused_unfocused" / split / (
        "{}.{}".format(_slug(dataset), args.format)
    )
    _save_figure(fig, file_path, args.dpi)
    return file_path


def create_plots(args):
    if (
            args.y_min is not None
            and args.y_max is not None
            and args.y_min >= args.y_max):
        raise ValueError("--y-min must be less than --y-max")

    run_data = _discover_run(args.run_directory, args.metric)
    task_indices = {
        task: index for index, task in enumerate(run_data.task_order)
    }
    recognized_records = [
        record
        for records in run_data.records.values()
        for record in records
        if record.evaluation_point.startswith("after_")
        and _task_name(record) in task_indices
    ]
    if not recognized_records:
        raise ValueError("No recognized after-evaluation rows found.")

    max_cycle = max(record.cycle for record in recognized_records)
    y_limits = _y_limits(run_data, args.y_min, args.y_max)
    output_directory = (
        args.output_root.expanduser().resolve()
        / run_data.run_directory.name
    )

    output_paths = []
    for dataset in run_data.datasets:
        for split in SPLITS:
            output_paths.append(
                _plot_focused(
                    run_data,
                    dataset,
                    split,
                    output_directory,
                    args,
                    y_limits,
                    max_cycle,
                )
            )
        for split in SPLITS:
            output_paths.append(
                _plot_unfocused(
                    run_data,
                    dataset,
                    split,
                    output_directory,
                    args,
                    y_limits,
                    max_cycle,
                )
            )
            output_paths.append(
                _plot_focused_unfocused(
                    run_data,
                    dataset,
                    split,
                    output_directory,
                    args,
                    y_limits,
                    max_cycle,
                )
            )

    print("Run: {}".format(run_data.run_directory.name))
    print("Datasets: {}".format(", ".join(run_data.datasets)))
    print("Evaluation order: {}".format(" -> ".join(run_data.task_order)))
    print("Metric: {}".format(args.metric))
    print(
        "Wrote {} plots to {}".format(
            len(output_paths),
            output_directory,
        )
    )
    return output_paths


def main():
    create_plots(parse_args())


if __name__ == "__main__":
    main()
