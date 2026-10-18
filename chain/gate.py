"""Stage A gate (research plan 5.2): from three Ark+ frozen-probe runs and two SSL pilots, decide the main run's LR and epochs.

    python chain/gate.py --arm_d RUN --probe_a RUN --probe_b RUN --pilot_a SSL_OUT --pilot_b SSL_OUT --out chain \
        [--min_gain 0.003] [--max_epoch_minutes 90]

RUN is an Ark run directory (reads RUN/train.log). SSL_OUT is an SSL output directory (reads ssl_losses.csv, and args.json if present).
main_lr.txt is the chosen pilot's effective peak LR, an absolute value (args.json effective_peak_lr, else the highest lr in ssl_losses.csv);
main_lr_scaling.txt is always "absolute" (chain/main.sh passes --no_lr_scaling and never reads it).
main_epochs.txt is 40 if the chosen pilot's epoch_minutes <= --max_epoch_minutes (the pre-registered 90, unchanged), else 30. epoch_minutes is
measured on full-backbone epochs only: the step-weighted mean step_time of the CSV rows with epoch >= 1, times steps per epoch, over 60.
Exit 0: decision written (main_lr.txt, main_lr_scaling.txt, main_epochs.txt, gate_report.md); a STOP left in --out by an earlier run is removed.
Exit 2: STOP, neither pilot beats arm D (STOP and gate_report.md written); main_*.txt left in --out by an earlier run are removed.
Exit 1: unreadable or malformed input, a bad threshold, or bad usage (nothing is written, nothing in --out is touched).
Pure stdlib, Python 3.10.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from decimal import Decimal

# VinDrCXR is left out: its val list is its test list, so the engine skips its validation pass and logs no line for it.
DATASETS = ("ChestXray14", "CheXpert", "MIMIC")
EPS = 1e-9        # slack on float >= comparisons (the log carries 6 decimals)
FIRST_FULL_EPOCH = 1   # epoch 0 trains the heads only (--freeze_backbone_epochs 1) and holds the first-step compile row: no guide to the main run's epochs
DECISION_FILES = ("main_lr.txt", "main_lr_scaling.txt", "main_epochs.txt")
MAX_MIN_GAIN = 0.1   # largest accepted --min_gain (10 pt of AUROC): anything above is a typo, not a policy

# base/engine.py _evaluate_validation_sets logs these two kinds of line through the "ark" logger
# ("%(asctime)s %(levelname)s %(message)s"):
#   Evaluating student and teacher validation sets at {point} in Cycle {cycle}
#   Validation {point} {dataset}: student_loss={:.6f} teacher_loss={:.6f} student_mean_auroc={:.6f} teacher_mean_auroc={:.6f}
# point is after_<training dataset> (cyclic) or after_joint. Only the header line carries the cycle. The validation pattern
# tolerates extra fields; it cannot match "Validation summary ..." or the "Test ..." lines.
HEADER_RE = re.compile(r"Evaluating student and teacher validation sets at (after_\S+) in Cycle (\d+)\b")
VALIDATION_RE = re.compile(r"Validation (after_\S+) (\S+): .*?\bteacher_mean_auroc=(\S+)")


class GateError(Exception):
    """Unreadable or malformed input; main() prints it and exits 1."""


def finite(value, where):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not math.isfinite(number):
        raise GateError("{}: {!r} is not a finite number".format(where, value))
    return number


def probe_scores(run_dir):
    """{cycle: (last validation point of the cycle, mean teacher_mean_auroc over DATASETS)} from RUN/train.log."""
    path = os.path.join(run_dir, "train.log")
    last_pass = {}   # cycle -> (point, {dataset: teacher_mean_auroc text}); a later pass of the same cycle replaces an earlier one
    current = None
    seen = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as log_file:
            for number, line in enumerate(log_file, 1):
                header = HEADER_RE.search(line)
                if header:
                    current = (header.group(1), {})
                    last_pass[int(header.group(2))] = current
                    continue
                match = VALIDATION_RE.search(line)
                if match:
                    point, dataset, value = match.groups()
                    if current is None or current[0] != point:
                        raise GateError(
                            "{}:{}: validation line for {} at {} without an 'Evaluating ... validation sets at {} in Cycle N' line before it"
                            .format(path, number, dataset, point, point))
                    current[1][dataset] = value
                    seen += 1
    except OSError as error:
        raise GateError("cannot read {}: {}".format(path, error))
    if not seen:
        raise GateError("{}: no 'Validation after_<dataset> <dataset>: ... teacher_mean_auroc=' lines".format(path))
    scores = {}
    for cycle, (point, values) in sorted(last_pass.items()):
        means = []
        for dataset in DATASETS:
            if dataset not in values:
                raise GateError("{}: cycle {} ({}): no validation line for {}".format(path, cycle, point, dataset))
            means.append(finite(values[dataset], "{}: cycle {} ({}) {} teacher_mean_auroc".format(path, cycle, point, dataset)))
        scores[cycle] = (point, sum(means) / len(means))
    return scores


def best_cycle(scores):
    """(probe score, its cycle): the max over cycles, the earliest cycle on a tie."""
    cycle = max(scores, key=lambda c: (scores[c][1], -c))
    return scores[cycle][1], cycle


def read_pilot(out_dir):
    """Facts from an SSL output dir: ssl_losses.csv (required) and args.json (optional)."""
    csv_path = os.path.join(out_dir, "ssl_losses.csv")
    rows = []   # (epoch, step, lr, step_time, steps that step_time averages over)
    previous = None
    try:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = [c for c in ("step", "epoch", "lr", "step_time") if c not in (reader.fieldnames or [])]
            if missing:
                raise GateError("{}: missing column(s) {}".format(csv_path, ", ".join(missing)))
            for row in reader:
                if row["step"] == "step":   # header repeated by a writer that appends on a re-run
                    continue
                try:
                    values = (int(float(row["epoch"])), int(float(row["step"])), float(row["lr"]), float(row["step_time"]))
                    if not (math.isfinite(values[2]) and math.isfinite(values[3])):
                        raise ValueError("lr or step_time not finite")
                except (TypeError, ValueError, OverflowError):
                    raise GateError("{}:{}: unreadable step/epoch/lr/step_time in {}".format(csv_path, reader.line_num, dict(row)))
                # the trainer's step_time is the mean over the steps since the previous row (its timer restarts at each epoch, and each
                # epoch's last step is always a row), so the gap to the previous row is the number of steps the row stands for
                if previous is not None and values[1] <= previous:
                    raise GateError("{}:{}: step {} does not increase (the row before is step {})".format(csv_path, reader.line_num, values[1], previous))
                rows.append(values + (1 if previous is None else values[1] - previous,))
                previous = values[1]
    except OSError as error:
        raise GateError("cannot read {}: {}".format(csv_path, error))
    if not rows:
        raise GateError("{}: no data rows".format(csv_path))
    max_lr = max(row[2] for row in rows)
    if max_lr <= 0:
        raise GateError("{}: max lr is {}, expected a positive learning rate".format(csv_path, max_lr))
    first = min(row[0] for row in rows)
    steps = [row[1] for row in rows if row[0] == first]
    steps_per_epoch = max(steps) - min(steps) + 1   # = max(step | epoch 0) + 1 when the CSV starts at step 0
    full = [(row[3], row[4]) for row in rows if row[0] >= FIRST_FULL_EPOCH]
    if not full:
        raise GateError("{}: no rows with epoch >= {}: the pilot did not reach epoch {} (epoch_minutes is measured on full-backbone epochs only)"
                        .format(csv_path, FIRST_FULL_EPOCH, FIRST_FULL_EPOCH + 1))
    epoch_minutes = sum(step_time * steps for step_time, steps in full) / sum(steps for _, steps in full) * steps_per_epoch / 60

    args_path = os.path.join(out_dir, "args.json")
    args = {}
    if os.path.exists(args_path):
        try:
            with open(args_path, encoding="utf-8") as args_file:
                args = json.load(args_file)
        except (OSError, ValueError) as error:
            raise GateError("cannot read {}: {}".format(args_path, error))
        if not isinstance(args, dict):
            raise GateError("{}: expected a JSON object".format(args_path))
    peak = None
    if args.get("effective_peak_lr") is not None:
        peak = finite(args["effective_peak_lr"], "{}: effective_peak_lr".format(args_path))
        if peak <= 0:
            raise GateError("{}: effective_peak_lr is {}, expected a positive learning rate".format(args_path, peak))
    return {
        "max_lr": max_lr,             # highest lr logged: already an effective (post-scaling) value, but below the peak if the run ended inside warmup
        "effective_peak_lr": peak,    # the trainer's args.json: --lr after its scaling rule, the schedule's peak; None: no args.json or no such key
        "steps_per_epoch": steps_per_epoch,
        "epoch_minutes": epoch_minutes,
    }


def decide(arm_d, score_a, score_b, lr_a, lr_b, min_gain):
    """The pre-registered rule (research plan 5.2): (chosen 'pilot_a' / 'pilot_b' or None for STOP, the rule that fired)."""
    scores = {"pilot_a": score_a, "pilot_b": score_b}
    beats = [name for name in scores if scores[name] >= arm_d + min_gain - EPS]
    if not beats:
        return None, "neither pilot beats arm D by min_gain"
    if len(beats) == 1:
        return beats[0], "only {} beats arm D".format(beats[0])
    # both beat arm D: the higher LR wins unless it scores more than min_gain below the other; an LR tie goes to pilot_a, the published LR
    lr_tie = math.isclose(lr_a, lr_b, rel_tol=1e-9)
    high, low = ("pilot_b", "pilot_a") if lr_b > lr_a and not lr_tie else ("pilot_a", "pilot_b")
    how = "an LR tie, taken as the published LR" if lr_tie else "the higher LR"
    if scores[high] >= scores[low] - min_gain - EPS:
        return high, "both beat arm D; {} ({}) is within min_gain of {}".format(high, how, low)
    return low, "both beat arm D; {} ({}) scores more than min_gain below {}, so the higher-scoring {} is chosen".format(high, how, low, low)


def plain(number):
    """Decimal text without an exponent, from the shortest repr (1e-05 -> 0.00001)."""
    return format(Decimal(repr(number)), "f")


def cycles_text(scores):
    return ", ".join("c{}={:.6f}".format(cycle, score) for cycle, (_, score) in sorted(scores.items()))


def evaluate(args):
    """Read everything and decide: (status 'DECISION' or 'STOP', report text, {file name: content} of the decision files)."""
    runs = {"arm D": probe_scores(args.arm_d), "probe_a": probe_scores(args.probe_a), "probe_b": probe_scores(args.probe_b)}
    best = {name: best_cycle(scores) for name, scores in runs.items()}
    pilots = {"pilot_a": read_pilot(args.pilot_a), "pilot_b": read_pilot(args.pilot_b)}
    for pilot in pilots.values():   # the peak LR a pilot ran at: the trainer's effective_peak_lr, else the highest lr it logged
        pilot["lr"], pilot["lr_source"] = ((pilot["max_lr"], "max observed lr") if pilot["effective_peak_lr"] is None
                                           else (pilot["effective_peak_lr"], "args.json effective_peak_lr"))
    arm_d = best["arm D"][0]
    chosen, rule = decide(arm_d, best["probe_a"][0], best["probe_b"][0], pilots["pilot_a"]["lr"], pilots["pilot_b"]["lr"], args.min_gain)

    lines = [
        "# Stage A gate report",
        "",
        "Probe score = max over cycles of the mean teacher_mean_auroc over {} at the cycle's last validation point (VinDrCXR excluded: its val list is its test list).".format(", ".join(DATASETS)),
        "Pre-registered rule (research plan 5.2): a pilot beats arm D if its probe score >= arm D + {:g} ({:g} pt).".format(args.min_gain, args.min_gain * 100),
        "epoch_minutes = step-weighted mean step_time over the pilot's rows with epoch >= {} (full-backbone epochs only) x steps/epoch / 60; "
        "40 epochs if <= {:g} min, else 30.".format(FIRST_FULL_EPOCH, args.max_epoch_minutes),
        "",
        "| run | run dir | per-cycle scores | validation point(s) | probe score (cycle) |",
        "|---|---|---|---|---|",
    ]
    for name, run_dir in (("arm D", args.arm_d), ("probe_a", args.probe_a), ("probe_b", args.probe_b)):
        score, cycle = best[name]
        points = ", ".join(sorted({point for point, _ in runs[name].values()}))
        lines.append("| {} | {} | {} | {} | {:.6f} (c{}) |".format(name, run_dir, cycles_text(runs[name]), points, score, cycle))
    lines += [
        "",
        "| pilot | out dir | peak lr | peak lr from | max observed lr | steps/epoch | epoch_minutes | probe score | gain over arm D | beats arm D |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, out_dir, probe in (("pilot_a", args.pilot_a, "probe_a"), ("pilot_b", args.pilot_b, "probe_b")):
        pilot = pilots[name]
        gain = best[probe][0] - arm_d
        lines.append("| {} | {} | {} | {} | {} | {} | {:.1f} | {:.6f} | {:+.6f} | {} |".format(
            name, out_dir, plain(pilot["lr"]), pilot["lr_source"], plain(pilot["max_lr"]), pilot["steps_per_epoch"], pilot["epoch_minutes"],
            best[probe][0], gain, "yes" if gain >= args.min_gain - EPS else "no"))
    lines.append("")

    if chosen is None:
        lines += ["Decision: STOP ({}). No main_lr.txt, main_lr_scaling.txt or main_epochs.txt written.".format(rule)]
        return "STOP", "\n".join(lines) + "\n", {}

    pilot = pilots[chosen]
    notes = ["WARNING: {} has no effective_peak_lr in its args.json (no args.json, or no such key), so its peak lr, for the higher-lr comparison and "
             "if it wins for main_lr.txt, is the max observed lr; that is below the effective peak if the pilot ended inside warmup.".format(name)
             for name, other in pilots.items() if other["effective_peak_lr"] is None]
    epochs = 40 if pilot["epoch_minutes"] <= args.max_epoch_minutes else 30
    lines += [
        "Decision: {}. Rule that fired: {}.".format(chosen, rule),
        "- main_lr.txt = {} ({}; an absolute peak lr, main.sh passes --no_lr_scaling)".format(plain(pilot["lr"]), pilot["lr_source"]),
        "- main_lr_scaling.txt = absolute (always; main.sh ignores this file)",
        "- main_epochs.txt = {} (epoch_minutes {:.1f} {} {:g})".format(epochs, pilot["epoch_minutes"], "<=" if epochs == 40 else ">", args.max_epoch_minutes),
    ] + notes
    # main.sh always runs --no_lr_scaling and reads main_lr.txt as an absolute peak lr; main_lr_scaling.txt is only a record, so it never says "scaled"
    files = {"main_lr.txt": plain(pilot["lr"]) + "\n", "main_lr_scaling.txt": "absolute\n", "main_epochs.txt": "{}\n".format(epochs)}
    return "DECISION", "\n".join(lines) + "\n", files


def check_thresholds(args):
    """Reject values that would silently force STOP or 30 epochs: nan makes every comparison false, inf and negatives are nonsense."""
    if not (math.isfinite(args.min_gain) and 0 <= args.min_gain <= MAX_MIN_GAIN):
        raise GateError("--min_gain must be a finite number from 0 to {:g}, got {}".format(MAX_MIN_GAIN, args.min_gain))
    if not (math.isfinite(args.max_epoch_minutes) and args.max_epoch_minutes > 0):
        raise GateError("--max_epoch_minutes must be a finite number above 0, got {}".format(args.max_epoch_minutes))


def remove_stale(out_dir, names):
    """Delete files an earlier run left in out_dir that the new result contradicts."""
    for name in names:
        try:
            os.remove(os.path.join(out_dir, name))
        except FileNotFoundError:
            pass


def write(out_dir, name, text):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as out_file:
        out_file.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, text in (("arm_d", "arm D (stock weights) frozen-probe run dir"), ("probe_a", "frozen-probe run dir on pilot_a's weights"),
                       ("probe_b", "frozen-probe run dir on pilot_b's weights"), ("pilot_a", "SSL output dir of pilot_a (published LR)"),
                       ("pilot_b", "SSL output dir of pilot_b"), ("out", "directory for the decision files (chain)")):
        parser.add_argument("--" + name, required=True, help=text)
    parser.add_argument("--min_gain", type=float, default=0.003, help="AUROC gain over arm D that counts as beating it (default 0.003)")
    parser.add_argument("--max_epoch_minutes", type=float, default=90.0, help="longest epoch that still allows 40 epochs, else 30 (default 90)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:   # argparse exits 2 on bad usage, but 2 means STOP here
        return 1 if stop.code else 0
    try:
        check_thresholds(args)   # before any run is read
        status, report, files = evaluate(args)
        os.makedirs(args.out, exist_ok=True)
        # the new result is fully computed: only now drop what an earlier run in this --out left that contradicts it
        remove_stale(args.out, ("STOP",) if status == "DECISION" else DECISION_FILES)
        write(args.out, "gate_report.md", report)
        if status == "STOP":
            write(args.out, "STOP", "STOP: neither pilot beat arm D by {:g}; the chain stops before main. Remove this file to let the driver continue.\n\n{}".format(args.min_gain, report))
        for name, text in files.items():
            write(args.out, name, text)
    except (GateError, OSError) as error:
        print("gate.py: error: {}".format(error), file=sys.stderr)
        return 1
    print(report, end="")
    return 2 if status == "STOP" else 0


if __name__ == "__main__":
    sys.exit(main())
