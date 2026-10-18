"""Tests for chain/gate.py and the chain driver. Run from the repo root: python3 chain/test_gate.py"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = os.path.join(HERE, "gate.py")
DRIVER = os.path.join(ROOT, "run_d1_8_chain_gaudi.sh")
ENGINE = os.path.join(ROOT, "base", "engine.py")
sys.path.insert(0, HERE)
import gate  # noqa: E402

# Log formats copied from base/engine.py (_evaluate_validation_sets, _evaluate_test_sets); test_engine_log_formats_unchanged
# fails if the engine drifts from the first two.
HEADER = "Evaluating student and teacher validation sets at {} in Cycle {}"
VALIDATION = ("Validation {} {}: student_loss={:.6f} teacher_loss={:.6f} "
              "student_mean_auroc={:.6f} teacher_mean_auroc={:.6f}")
SUMMARY = ("Validation summary {} Cycle={}: student_losses={} "
           "teacher_losses={} student_mean_aurocs={} teacher_mean_aurocs={}")
SKIPPED = ("Skipping separate validation evaluation for {} at {}; "
           "validation and test samples are identical")
TEST_HEADER = "Evaluating student and teacher test sets at {} in Cycle {}"
TEST_LINE = "Test {} {}: student_mean_auroc={:.6f} teacher_mean_auroc={:.6f}"
LOG_PREFIX = "2026-10-08 12:00:00 INFO "   # the "ark" logger: %(asctime)s %(levelname)s %(message)s
ORDER = ["VinDrCXR", "ChestXray14", "MIMIC", "CheXpert"]   # all4 training order, so the last point of a cycle is after_CheXpert
SPREAD = {"ChestXray14": -0.01, "CheXpert": 0.0, "MIMIC": 0.01}   # sums to zero: the three-dataset mean is exactly the requested one
# args.json as base/ssl_pretrain.py writes it (vars(args) plus effective_*): effective_peak_lr is --lr after the sqrt_wrt_1024 factor, or --lr
# itself under --no_lr_scaling. Pilot A here: --lr 2e-4 x 3.5 = 7e-4; pilot B: absolute 4e-4.
ARGS_A = {"lr": 2e-4, "no_lr_scaling": False, "lr_scaling_factor": 3.5, "effective_peak_lr": 7e-4, "effective_end_lr": 3.5e-6, "epochs": 40}
ARGS_B = {"lr": 4e-4, "no_lr_scaling": True, "lr_scaling_factor": 1.0, "effective_peak_lr": 4e-4, "effective_end_lr": 1e-6, "epochs": 40}
COLUMNS = "step,epoch,dino_global,dino_local,koleo,ibot,gram,total,lr,wd,momentum,teacher_temp,gram_weight,step_time,data_time".split(",")


def make_probe(run_dir, cycle_means, mutate=None):
    """RUN/train.log whose last point of cycle n has the three-dataset mean cycle_means[n-1]. Everything that must not count
    (earlier points of a cycle, Shenzhen, the Test lines) carries a decoy 0.99 that would inflate the score if it leaked in."""
    os.makedirs(run_dir, exist_ok=True)
    lines = ["Cycles: {}".format(len(cycle_means))]
    for cycle, mean in enumerate(cycle_means, 1):
        for trained in ORDER:
            point = "after_" + trained
            lines += [TEST_HEADER.format(point, cycle)] + [TEST_LINE.format(point, d, 0.99, 0.99) for d in ORDER]
            lines.append(HEADER.format(point, cycle))
            for dataset in ORDER + ["Shenzhen"]:
                if dataset == "VinDrCXR":   # shared val/test split: the engine logs this instead of a validation line
                    lines.append(SKIPPED.format(dataset, point))
                    continue
                final = trained == ORDER[-1] and dataset in SPREAD
                teacher = mean + SPREAD[dataset] if final else 0.99
                lines.append(VALIDATION.format(point, dataset, 0.4, 0.4, teacher - 0.02, teacher))
            lines.append(SUMMARY.format(point, cycle, "[]", "[]", "[]", "[]"))
    if mutate:
        lines = mutate(lines)
    with open(os.path.join(run_dir, "train.log"), "w") as log_file:
        log_file.write("".join(LOG_PREFIX + line + "\n" for line in lines))


def write_losses(out_dir, rows):
    """ssl_losses.csv in the trainer's column contract from (step, epoch, lr, step_time) tuples."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ssl_losses.csv"), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(COLUMNS)
        for step, epoch, lr, step_time in rows:
            row = dict.fromkeys(COLUMNS, 0.1)
            row.update(step=step, epoch=epoch, lr=lr, step_time=step_time)
            writer.writerow([row[c] for c in COLUMNS])


def make_pilot(out_dir, peak_lr, step_time=30.0, steps_per_epoch=100, epochs=2, frozen_time=None, first_step_time=None, args=None):
    """SSL out dir: a CSV row for every step (lr ramps linearly to peak_lr at the last step; epoch 0, the frozen-backbone epoch, takes
    frozen_time per step and step 0, the compile step, first_step_time, when given), and args.json when args is given."""
    total = steps_per_epoch * epochs

    def seconds(step):
        if step == 0 and first_step_time is not None:
            return first_step_time
        return frozen_time if step < steps_per_epoch and frozen_time is not None else step_time

    write_losses(out_dir, [(step, step // steps_per_epoch, peak_lr * min(1.0, (step + 1) / total), seconds(step)) for step in range(total)])
    args_path = os.path.join(out_dir, "args.json")
    if args is not None:
        with open(args_path, "w") as args_file:
            json.dump(args, args_file)
    elif os.path.exists(args_path):
        os.remove(args_path)


class GateTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.out = os.path.join(self.tmp, "chain")

    def path(self, name):
        return os.path.join(self.tmp, name)

    def run_gate(self, arm_d=(0.80,), probe_a=(0.805,), probe_b=(0.805,), pilot_a=None, pilot_b=None, extra=()):
        """Build the five run dirs and run gate.py. Defaults: pilot_a (--lr 2e-4 scaled to an effective peak of 7e-4) has the higher peak
        lr than pilot_b (absolute 4e-4), and both pilots' logged lr reaches its peak; both beat arm D by 0.005."""
        pilot_a = pilot_a or dict(peak_lr=7e-4, args=ARGS_A)
        pilot_b = pilot_b or dict(peak_lr=4e-4, args=ARGS_B)
        make_probe(self.path("armD"), arm_d)
        make_probe(self.path("probe_a"), probe_a)
        make_probe(self.path("probe_b"), probe_b)
        make_pilot(self.path("pilot_a"), **pilot_a)
        make_pilot(self.path("pilot_b"), **pilot_b)
        argv = [sys.executable, GATE, "--arm_d", self.path("armD"), "--probe_a", self.path("probe_a"), "--probe_b", self.path("probe_b"),
                "--pilot_a", self.path("pilot_a"), "--pilot_b", self.path("pilot_b"), "--out", self.out, *extra]
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)

    def read(self, name):
        with open(os.path.join(self.out, name)) as out_file:
            return out_file.read()

    def assert_decision(self, proc, lr, epochs="40"):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # main_lr_scaling.txt is always "absolute": chain/main.sh passes --no_lr_scaling and never reads it
        self.assertEqual((self.read("main_lr.txt"), self.read("main_lr_scaling.txt"), self.read("main_epochs.txt")),
                         (lr + "\n", "absolute\n", epochs + "\n"))
        self.assertEqual(self.read("gate_report.md"), proc.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.out, "STOP")))

    def assert_nothing_written(self, proc, code):
        self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
        for name in ("main_lr.txt", "main_lr_scaling.txt", "main_epochs.txt"):
            self.assertFalse(os.path.exists(os.path.join(self.out, name)), name)

    def test_a_both_beat_arm_d_higher_lr_pilot_a_wins(self):
        # pilot_a: effective peak 7e-4 (--lr 2e-4 x 3.5) > 4e-4, scores within 0.003 of pilot_b; main_lr is that effective peak, an absolute lr
        self.assert_decision(self.run_gate(probe_a=(0.805,), probe_b=(0.806,)), "0.0007")

    def test_a2_higher_lr_pilot_too_far_behind_loses_to_the_higher_scorer(self):
        self.assert_decision(self.run_gate(probe_a=(0.805,), probe_b=(0.812,)), "0.0004")

    def test_a3_higher_lr_is_pilot_b(self):
        proc = self.run_gate(pilot_b=dict(peak_lr=9e-4, args=dict(ARGS_B, lr=9e-4, effective_peak_lr=9e-4)))
        self.assert_decision(proc, "0.0009")

    def test_a4_lr_tie_goes_to_pilot_a(self):
        proc = self.run_gate(pilot_b=dict(peak_lr=7e-4, args=dict(ARGS_B, lr=7e-4, effective_peak_lr=7e-4)))
        self.assert_decision(proc, "0.0007")

    def test_b_only_pilot_b_beats_arm_d(self):
        self.assert_decision(self.run_gate(probe_a=(0.801,), probe_b=(0.806,)), "0.0004")

    def test_b2_only_pilot_a_beats_arm_d(self):
        self.assert_decision(self.run_gate(probe_a=(0.806,), probe_b=(0.801,)), "0.0007")

    def test_c_neither_beats_arm_d_writes_stop_and_exits_2(self):
        proc = self.run_gate(probe_a=(0.801,), probe_b=(0.802,))
        self.assert_nothing_written(proc, 2)
        stop = self.read("STOP")
        self.assertIn("STOP", stop)
        self.assertIn("arm D", stop)
        self.assertIn("pilot_a", stop)
        self.assertEqual(self.read("gate_report.md"), proc.stdout)

    def test_gain_of_exactly_min_gain_beats_arm_d(self):
        # 0.683 - 0.680 is 0.00299999... in floats (0.667 vs 0.670 and 73 more 3-decimal pairs too); the rule says >=, so it must count
        self.assert_decision(self.run_gate(arm_d=(0.68,), probe_a=(0.683,), probe_b=(0.60,)), "0.0007")

    def test_d_epochs_follow_epoch_minutes(self):
        # epoch_minutes = step-weighted mean step_time over the rows with epoch >= 1, * steps_per_epoch / 60; 40 epochs if <= max, else 30
        cases = [
            ("40", dict(step_time=30.0)),                                  # 100 steps * 30 s = 50 min
            ("30", dict(step_time=60.0)),                                  # 100 min
            ("40", dict(step_time=54.0)),                                  # exactly 90.0 min: <= keeps 40
            ("30", dict(step_time=54.5)),                                  # 90.8 min
            ("30", dict(step_time=20.0, steps_per_epoch=300)),             # steps_per_epoch comes from epoch 0: 100 min
            ("30", dict(step_time=60.0, frozen_time=1.0)),                 # a fast frozen epoch 0 would have averaged this down to 50.8 min
            ("40", dict(step_time=30.0, first_step_time=1e6)),             # a compile row would have pushed this to thousands of minutes
        ]
        for expected, pilot in cases:
            with self.subTest(expected=expected, pilot=pilot):
                proc = self.run_gate(probe_a=(0.81,), probe_b=(0.80,),
                                     pilot_a=dict(peak_lr=7e-4, args=ARGS_A, **pilot))
                self.assert_decision(proc, "0.0007", expected)

    def test_epoch_minutes_ignores_the_frozen_epoch_and_its_compile_row(self):
        make_pilot(self.path("clean"), 7e-4, step_time=30.0, epochs=3)
        make_pilot(self.path("noisy"), 7e-4, step_time=30.0, epochs=3, frozen_time=1.0, first_step_time=1e6)
        for name in ("clean", "noisy"):
            pilot = gate.read_pilot(self.path(name))
            self.assertEqual(pilot["steps_per_epoch"], 100)
            self.assertAlmostEqual(pilot["epoch_minutes"], 50.0, places=9)   # 100 steps * 30 s / 60

    def test_epoch_minutes_weights_each_row_by_its_step_gap(self):
        # the trainer logs every print_freq steps (and each epoch's last step), so a row stands for the steps since the previous row,
        # across an epoch start too: epoch 1's first row (step 100) is 1 step, then 49 and 50
        write_losses(self.path("p"), [(0, 0, 1e-5, 500.0), (49, 0, 1e-4, 1.0), (99, 0, 2e-4, 1.0),
                                      (100, 1, 3e-4, 100.0), (149, 1, 4e-4, 10.0), (199, 1, 5e-4, 20.0)])
        pilot = gate.read_pilot(self.path("p"))
        self.assertEqual(pilot["steps_per_epoch"], 100)
        self.assertAlmostEqual(pilot["epoch_minutes"], (100 * 1 + 10 * 49 + 20 * 50) / 100 * 100 / 60, places=9)   # 26.5 min (unweighted: 72.2)
        # a CSV that starts mid-run must not over-weight its first row (it weighs 1, not step + 1): same 26.5 min from epoch 1 alone
        write_losses(self.path("q"), [(100, 1, 3e-4, 100.0), (149, 1, 4e-4, 10.0), (199, 1, 5e-4, 20.0)])
        self.assertAlmostEqual(gate.read_pilot(self.path("q"))["epoch_minutes"], pilot["epoch_minutes"], places=9)

    def test_pilot_that_did_not_reach_epoch_two_exits_1(self):
        proc = self.run_gate(pilot_a=dict(peak_lr=7e-4, args=ARGS_A, epochs=1))
        self.assert_nothing_written(proc, 1)
        self.assertIn("did not reach epoch 2", proc.stderr)

    def test_steps_that_do_not_increase_are_refused(self):
        write_losses(self.path("p"), [(0, 0, 1e-5, 1.0), (1, 0, 1e-4, 1.0), (1, 1, 2e-4, 1.0)])
        with self.assertRaisesRegex(gate.GateError, "step 1 does not increase"):
            gate.read_pilot(self.path("p"))

    def test_max_epoch_minutes_is_a_parameter(self):
        self.assert_decision(self.run_gate(probe_b=(0.80,), extra=("--max_epoch_minutes", "40")), "0.0007", "30")

    def test_e_run_without_validation_lines_exits_1(self):
        for name, mutate in (("none", lambda lines: [l for l in lines if not l.startswith("Validation after_")]),
                             ("summary only", lambda lines: [l for l in lines if l.startswith(("Validation summary", "Evaluating"))])):
            with self.subTest(name):
                make_probe(self.path(name), (0.805,), mutate=mutate)
                proc = self.run_gate(extra=("--probe_a", self.path(name)))   # argparse keeps the last --probe_a
                self.assert_nothing_written(proc, 1)
                self.assertIn("no 'Validation after_", proc.stderr)

    def test_unusable_inputs_exit_1_and_write_nothing(self):
        make_probe(self.path("broken"), (0.8,), mutate=lambda lines: [l.replace("teacher_mean_auroc=0.80", "teacher_mean_auroc=nan") for l in lines])
        make_probe(self.path("no_mimic"), (0.8,), mutate=lambda lines: [l for l in lines if " MIMIC:" not in l])
        os.makedirs(self.path("empty"))
        for name, expected in (("broken", "not a finite number"), ("no_mimic", "no validation line for MIMIC"), ("empty", "cannot read")):
            with self.subTest(name):
                proc = self.run_gate(extra=("--probe_b", self.path(name)))   # argparse keeps the last --probe_b
                self.assert_nothing_written(proc, 1)
                self.assertIn(expected, proc.stderr)

    def test_missing_args_exit_1_not_2(self):
        proc = subprocess.run([sys.executable, GATE, "--arm_d", self.tmp], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)

    def test_decision_after_stop_clears_the_stale_stop(self):
        self.assert_nothing_written(self.run_gate(probe_a=(0.801,), probe_b=(0.802,)), 2)
        self.assertTrue(os.path.exists(os.path.join(self.out, "STOP")))
        self.assert_decision(self.run_gate(), "0.0007")   # also asserts that STOP is gone

    def test_stop_after_a_decision_clears_the_stale_decision_files(self):
        self.assert_decision(self.run_gate(), "0.0007")
        proc = self.run_gate(probe_a=(0.801,), probe_b=(0.802,))
        self.assert_nothing_written(proc, 2)   # main_*.txt are gone
        self.assertTrue(os.path.exists(os.path.join(self.out, "STOP")))
        self.assertEqual(self.read("gate_report.md"), proc.stdout)

    def test_error_leaves_earlier_results_untouched(self):
        for name, scores in (("after a decision", {}), ("after a stop", dict(probe_a=(0.801,), probe_b=(0.802,)))):
            with self.subTest(name):
                self.run_gate(**scores)
                before = {n: self.read(n) for n in sorted(os.listdir(self.out))}
                proc = self.run_gate(extra=("--probe_b", self.path("missing")))   # argparse keeps the last --probe_b
                self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                self.assertEqual({n: self.read(n) for n in sorted(os.listdir(self.out))}, before)

    def test_bad_thresholds_exit_1_before_any_run_is_read(self):
        bad = {"--min_gain": ("nan", "inf", "-inf", "-0.001", "0.11", "0.5"),
               "--max_epoch_minutes": ("nan", "inf", "-inf", "-5", "0")}
        runs = [a for flag in ("--arm_d", "--probe_a", "--probe_b", "--pilot_a", "--pilot_b") for a in (flag, self.path("nowhere"))]
        for flag, values in bad.items():
            for value in values:
                with self.subTest(flag=flag, value=value):
                    argv = [sys.executable, GATE, *runs, "--out", self.out, flag + "=" + value]   # "=" so that -inf is a value, not a flag
                    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
                    self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                    self.assertIn(flag + " must be a finite number", proc.stderr)   # the threshold is reported, not the missing runs
                    self.assertFalse(os.path.exists(self.out))

    def test_valid_threshold_bounds_are_accepted(self):
        # --min_gain 0: a tie with arm D beats it; 0.1 is the largest accepted (0.90 - 0.80 is 0.0999... in floats)
        self.assert_decision(self.run_gate(arm_d=(0.80,), probe_a=(0.80,), probe_b=(0.70,), extra=("--min_gain=0",)), "0.0007")
        self.assert_decision(self.run_gate(arm_d=(0.80,), probe_a=(0.90,), probe_b=(0.70,), extra=("--min_gain=0.1",)), "0.0007")
        self.assert_decision(self.run_gate(extra=("--max_epoch_minutes=0.5",)), "0.0007", "30")

    def test_probe_score_is_last_point_of_three_datasets_max_over_cycles(self):
        make_probe(self.path("run"), (0.80, 0.83, 0.81))
        scores = gate.probe_scores(self.path("run"))
        self.assertEqual({c: point for c, (point, _) in scores.items()}, {1: "after_CheXpert", 2: "after_CheXpert", 3: "after_CheXpert"})
        for cycle, mean in zip((1, 2, 3), (0.80, 0.83, 0.81)):
            self.assertAlmostEqual(scores[cycle][1], mean, places=9)
        self.assertEqual(gate.best_cycle(scores)[1], 2)

    def test_main_lr_is_the_winners_effective_peak_not_the_configured_lr_or_the_logged_max(self):
        # --lr 2e-4 x 3.5 = 7e-4 effective; the CSV only reached 3e-4 because the pilot ended early in warmup
        proc = self.run_gate(probe_b=(0.80,), pilot_a=dict(peak_lr=3e-4, args=ARGS_A))
        self.assert_decision(proc, "0.0007")
        self.assertIn("(args.json effective_peak_lr;", proc.stdout)

    def test_effective_peak_lr_beats_the_logged_lr_in_the_higher_lr_comparison(self):
        # pilot_a stopped early in warmup (logged up to 4e-4 of an effective peak 9e-4); pilot_b logged 5e-4 = its peak. By the logged lr
        # pilot_b would be the higher one, by the effective peak pilot_a is. Equal scores, so the higher LR wins.
        proc = self.run_gate(pilot_a=dict(peak_lr=4e-4, args=dict(ARGS_A, effective_peak_lr=9e-4)),
                             pilot_b=dict(peak_lr=5e-4, args=dict(ARGS_B, effective_peak_lr=5e-4)))
        self.assert_decision(proc, "0.0009")

    def test_without_effective_peak_lr_the_max_logged_lr_stands_in_with_a_warning(self):
        # no args.json at all, an older one without the key, or null: the highest lr in ssl_losses.csv is the pilot's peak lr
        old_args = {k: v for k, v in ARGS_A.items() if not k.startswith("effective")}
        for name, args in (("no args.json", None), ("no effective_peak_lr", old_args), ("null", dict(ARGS_A, effective_peak_lr=None))):
            with self.subTest(name):
                proc = self.run_gate(probe_b=(0.80,), pilot_a=dict(peak_lr=7e-4, args=args), pilot_b=dict(peak_lr=4e-4, args=args))
                self.assert_decision(proc, "0.0007")
                self.assertIn("WARNING: pilot_a has no effective_peak_lr", proc.stdout)
                self.assertIn("WARNING: pilot_b has no effective_peak_lr", proc.stdout)
                self.assertIn("(max observed lr;", proc.stdout)
        proc = self.run_gate(probe_a=(0.80,), pilot_a=dict(peak_lr=7e-4), pilot_b=dict(peak_lr=4e-4))   # pilot_b wins: its own logged max
        self.assert_decision(proc, "0.0004")

    def test_unusable_effective_peak_lr_exits_1_and_writes_nothing(self):
        for bad in ("abc", "nan", 0, -0.001, [1]):
            with self.subTest(bad=bad):
                proc = self.run_gate(pilot_a=dict(peak_lr=7e-4, args=dict(ARGS_A, effective_peak_lr=bad)))
                self.assert_nothing_written(proc, 1)
                self.assertIn("effective_peak_lr", proc.stderr)

    def test_engine_log_formats_unchanged(self):
        if not os.path.exists(ENGINE):
            self.skipTest("base/engine.py not found")
        with open(ENGINE) as engine_file:
            source = re.sub(r'"\s*\n\s*"', "", engine_file.read())   # join implicitly concatenated string literals
        self.assertIn('"' + HEADER + '"', source)
        self.assertIn('"' + VALIDATION + '"', source)
        self.assertIn('"' + SUMMARY + '"', source)


class DriverTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.chain = os.path.join(self.root, "chain")
        os.makedirs(os.path.join(self.chain, "done"))

    def setup_chain(self, phases, scripts=(), done=()):
        with open(os.path.join(self.chain, "phases.txt"), "w") as phases_file:
            phases_file.write(phases)
        for phase in scripts:
            self.script(phase, "echo ran {}\n".format(phase))
        for phase in done:
            open(os.path.join(self.chain, "done", phase), "w").close()

    def script(self, phase, body):
        with open(os.path.join(self.chain, phase + ".sh"), "w") as script_file:
            script_file.write(body)

    def run_driver(self, *args, driver=DRIVER):
        return subprocess.run(["bash", driver, *args], cwd=self.root, capture_output=True, text=True, timeout=60)

    def tree(self):   # every file, and (through the empty name) every directory
        return sorted(os.path.join(d, f) for d, _, files in os.walk(self.root) for f in files + [""])

    def messages(self, proc):
        """stdout lines without their timestamp (and the [dry-run] tag); every line must carry the timestamp."""
        lines = proc.stdout.splitlines()
        for line in lines:
            self.assertRegex(line, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ")
        return [re.sub(r"^\S+ (\[dry-run\] )?", "", line) for line in lines]

    def test_dry_run_prints_the_plan_and_changes_nothing(self):
        self.setup_chain("# plan\n\narmD\r\n  pilot_a   # trailing comment\nprobe_a\ngate", scripts=("armD", "pilot_a", "probe_a"), done=("armD",))
        before = self.tree()
        proc = self.run_driver("--dry-run")
        self.assertEqual(proc.returncode, 1, proc.stderr)   # gate.sh does not exist, so the plan ends where the real run would
        self.assertEqual(self.messages(proc)[1:], [
            "phase armD: done marker found, skipping",
            "phase pilot_a: would run bash chain/pilot_a.sh >> chain/log/pilot_a.out 2>&1",
            "phase probe_a: would run bash chain/probe_a.sh >> chain/log/probe_a.out 2>&1",
            "phase gate: script missing, stopping",
        ])
        self.assertIn("[dry-run]", proc.stdout)
        self.assertEqual(self.tree(), before)

    def test_dry_run_completes_when_every_phase_is_planned(self):
        self.setup_chain("a\nb\n", scripts=("a", "b"))
        proc = self.run_driver("--dry-run")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(self.messages(proc)[1:], [
            "phase a: would run bash chain/a.sh >> chain/log/a.out 2>&1",
            "phase b: would run bash chain/b.sh >> chain/log/b.out 2>&1",
            "no phase left to run in chain/phases.txt, chain complete",
        ])

    def test_dry_run_stops_cleanly_on_a_stop_file(self):
        self.setup_chain("a\nb\n", scripts=("a", "b"), done=("a",))
        open(os.path.join(self.chain, "STOP"), "w").close()
        proc = self.run_driver("--dry-run")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(self.messages(proc)[1:], ["STOP file present (chain/STOP), stopping"])

    def test_unknown_flag_is_refused(self):
        self.setup_chain("a\n", scripts=("a",))
        proc = self.run_driver("--dryrun")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", proc.stderr)

    def stripped_driver(self):
        """The driver without its Sol-only lines (cd, card checks, busy check), so the real loop runs on any machine."""
        lines, replaced = [], 0
        with open(DRIVER) as driver_file:
            for line in driver_file.read().splitlines():
                code = line.strip()
                if code and not code.startswith("#") and re.search(r"cd /scratch/|gaudi_check_cards|gaudi_verify_busy", code):
                    line, replaced = line[:len(line) - len(line.lstrip())] + ":", replaced + 1
                lines.append(line)
        self.assertEqual(replaced, 4)   # cd, card check, busy-script existence check, busy check
        path = os.path.join(self.root, "driver.sh")
        with open(path, "w") as out_file:
            out_file.write("\n".join(lines) + "\n")
        return path

    def test_real_loop_rereads_phases_and_records_times(self):
        self.setup_chain("one\ntwo\n", scripts=("one", "three"))
        self.script("two", "echo ran two\necho three >> chain/phases.txt\n")   # a phase appends the next one while the chain runs
        proc = self.run_driver(driver=self.stripped_driver())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for phase in ("one", "two", "three"):
            self.assertTrue(os.path.exists(os.path.join(self.chain, "done", phase)), phase)
        with open(os.path.join(self.chain, "times.csv")) as times_file:
            rows = list(csv.reader(times_file))
        self.assertEqual(rows[0], ["phase", "start_iso", "end_iso", "seconds", "exit_code"])
        self.assertEqual([(r[0], r[4]) for r in rows[1:]], [("one", "0"), ("two", "0"), ("three", "0")])
        with open(os.path.join(self.chain, "log", "three.out")) as log_file:
            self.assertIn("ran three", log_file.read())
        self.assertIn("phase three: done", proc.stdout)

    def test_real_loop_stops_at_a_failure_and_a_resubmit_continues(self):
        self.setup_chain("ok\nbad\nafter\n", scripts=("after",))
        self.script("ok", "echo x >> ok.count\n")
        self.script("bad", "echo failing\nexit 3\n")
        driver = self.stripped_driver()
        proc = self.run_driver(driver=driver)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("phase bad: FAILED with exit code 3", proc.stdout)
        self.assertTrue(os.path.exists(os.path.join(self.chain, "done", "ok")))
        self.assertFalse(os.path.exists(os.path.join(self.chain, "done", "bad")))
        self.assertFalse(os.path.exists(os.path.join(self.chain, "log", "after.out")))   # never continues past a failed phase
        with open(os.path.join(self.chain, "times.csv")) as times_file:
            self.assertEqual(times_file.read().splitlines()[-1].split(",")[0::4], ["bad", "3"])
        self.script("bad", "echo fixed\n")
        proc = self.run_driver(driver=driver)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(os.path.join(self.root, "ok.count")) as count_file:
            self.assertEqual(count_file.read(), "x\n")   # ok was not run again
        with open(os.path.join(self.chain, "log", "bad.out")) as log_file:
            text = log_file.read()
        self.assertEqual((text.count("driver: start bad"), "failing" in text, "fixed" in text), (2, True, True))   # appended, not overwritten

    def test_real_loop_honours_stop_and_missing_scripts(self):
        self.setup_chain("a\nb\n", scripts=("b",))
        self.script("a", "touch chain/STOP\n")
        driver = self.stripped_driver()
        proc = self.run_driver(driver=driver)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("STOP file present", proc.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.chain, "log", "b.out")))
        os.remove(os.path.join(self.chain, "STOP"))
        os.remove(os.path.join(self.chain, "b.sh"))
        proc = self.run_driver(driver=driver)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("phase b: script missing, stopping", proc.stdout)


if __name__ == "__main__":
    unittest.main()
