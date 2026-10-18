#!/usr/bin/env python3
"""Parallel-coordinates figure (Ark+ Nature 2025 Fig. 2a style) of frozen-backbone linear probes.

Reads <results_root>/<tag>/<dataset>.json as written by base/ark_linearprobe/run_linearprobe.py and writes a multi-page PDF
(one landscape page per dataset, then a summary table); the summary table is also printed. From the repository root:

  python plot/plot_linear_probes.py results --tags rad_dino=RAD-DINO stock_dinov3="DINOv3 stock" --reference VinDrCXR="Ark+ linear probe, Nature 2025 Fig. 3a"=96.19
  python plot/plot_linear_probes.py --selftest
"""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
try:
    from run_labels import dataset_display
except ImportError:  # pragma: no cover - supports ``python -m plot...``.
    from .run_labels import dataset_display

DATASETS = ("VinDrCXR", "ChestXray14", "MIMIC", "CheXpert")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F7F7F", "#000000")  # Okabe-Ito: distinct, colour-blind safe
FOOTER = "Protocol: Ark+ main_linearprobing.py as shipped on cached 1536-d (ViT-B) or 2048-d (ViT-L) CLS‖mean-patch features at 224 px; best epoch on test; mean ± s.d. over runs."
DAGGER = "† RAD-DINO's pretraining data overlaps this test set"
NEEDED = ("classes", "n", "max_images", "runs", "test_mAUC_mean", "test_mAUC_std", "test_AUC_mean", "test_AUC_std")  # max_images: a file without it cannot be shown to be full-list


def note(msg):
    print("plot_linear_probes: " + msg, file=sys.stderr)


def tag_arg(s):  # TAG=Label; the label defaults to the tag
    tag, _, label = s.partition("=")
    return tag, label.strip("\"' ") or tag


def reference_arg(s):  # DATASET=Label=VALUE, split on the first and the last "=" so the label may hold "=" itself
    ds, _, rest = s.partition("=")
    label, sep, value = rest.rpartition("=")
    try:
        value = float(value)
        if ds not in DATASETS or not sep or not np.isfinite(value):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: want DATASET=Label=VALUE, DATASET one of {', '.join(DATASETS)}")
    return ds, label.strip("\"' "), value


def parse_args(argv):
    p = argparse.ArgumentParser(description="Parallel-coordinates PDF of frozen-backbone linear probes (Ark+ Fig. 2a style).")
    p.add_argument("results_root", nargs="?", help="folder holding <tag>/<dataset>.json")
    p.add_argument("--tags", nargs="+", type=tag_arg, metavar="TAG=Label", help="tags to draw, in legend order (default: every tag directory, label = tag)")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_probes.pdf"))
    p.add_argument("--reference", nargs="+", type=reference_arg, default=[], metavar="DATASET=Label=VALUE", help="dashed grey line (percent) on the Mean axis of that dataset's page")
    p.add_argument("--title", default="", help="extra text under every page title")
    p.add_argument("--selftest", action="store_true", help="run the whole pipeline on synthetic JSONs, assert, and exit")
    a = p.parse_args(argv)
    if not a.selftest and not a.results_root:
        p.error("results_root is required")
    return a


def load(root, tag, ds):
    """The tag's JSON for ds, or None (with a note) when it was never written."""
    path = os.path.join(root, tag, ds + ".json")
    if not os.path.isfile(path):
        note(f"{path} missing: {tag} skipped on {ds}")
        return None
    with open(path) as f:
        d = json.load(f)
    bad = [k for k in NEEDED if k not in d] or [c for c in d["classes"] if c not in d["test_AUC_mean"] or c not in d["test_AUC_std"]]
    if bad or not 0 <= d["test_mAUC_mean"] <= 1:  # the runner writes fractions; a percent-valued file would be scaled by 100 twice
        sys.exit(f"plot_linear_probes: {path} is not a run_linearprobe.py result (bad: {bad or 'test_mAUC_mean outside [0, 1]'})")
    if d["max_images"] is not None:  # a smoke subset is not the full list: refused, never plotted
        sys.exit(f"plot_linear_probes: {tag}/{ds}: fitted on a smoke subset (max_images={d['max_images']}); smoke results are not plotted. Pass full-list tags only.")
    return d


def flagged(d):
    return bool((d.get("flags") or {}).get("rad_dino_pretraining_overlap"))


def dataset_page(ds, got, labels, refs, title, all_tags):
    """One landscape page: an axis per class (own min..max range over the tags, 1 point of padding) plus Mean; got = {tag: json} in legend order; all_tags = every tag in --tags order (colour and label side follow it, so they match across pages)."""
    first_tag, first = next(iter(got.items()))
    classes = first["classes"]
    for t, d in got.items():  # ponytail: tags that scored different columns are refused, not gapped; add NaN gaps if that must be drawn
        if d["classes"] != classes:
            sys.exit(f"plot_linear_probes: {ds}: tag {t} scored {d['classes']}, the first tag scored {classes}; the axes would not line up")
        if d["n"] != first["n"] or d["max_images"] != first["max_images"]:
            sys.exit(f"plot_linear_probes: {ds}: tag {t} has n {d['n']} and max_images {d['max_images']}, the first tag {first_tag} has n {first['n']} and max_images {first['max_images']}; the cohorts would not match")
    names = [c.replace("_", " ") for c in classes] + ["Mean"]
    m = len(names) - 1  # x of the Mean axis
    mean = np.array([[100 * d["test_AUC_mean"][c] for c in classes] + [100 * d["test_mAUC_mean"]] for d in got.values()])
    sd = np.array([[100 * d["test_AUC_std"][c] for c in classes] + [100 * d["test_mAUC_std"]] for d in got.values()])
    lo, hi = (mean - sd).min(0) - 1, (mean + sd).max(0) + 1  # error bars count toward the range, so none is cut off
    for _, v in refs:  # a reference line must fit on the Mean axis
        lo[m], hi[m] = min(lo[m], v - 1), max(hi[m], v + 1)
    span, xs = hi - lo, np.arange(m + 1)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.16)
    for j in xs:  # the vertical axes, each with its own ticks
        ax.plot([j, j], [0, 1], color="0.35", lw=0.8, zorder=1)
        for t in MaxNLocator(5).tick_values(lo[j], hi[j]):
            if lo[j] <= t <= hi[j]:
                y = (t - lo[j]) / span[j]
                ax.plot([j - 0.05, j], [y, y], color="0.35", lw=0.6, zorder=1)
                ax.text(j - 0.07, y, f"{t:.1f}", ha="right", va="center", fontsize=5.5, color="0.4", zorder=2)
    handles, texts, dagger = [], [], False
    for i, (t, d) in enumerate(got.items()):
        k = all_tags.index(t)  # full --tags position, not the position among this page's tags, so a tag keeps its colour on every page
        c, side = COLORS[k % len(COLORS)], (1 if k % 2 == 0 else -1)  # value labels alternate right/left by tag index
        y = (mean[i] - lo) / span
        handles.append(ax.errorbar(xs, y, yerr=sd[i] / span, fmt="o--", color=c, ms=4, lw=1.3, elinewidth=0.7, capsize=1.5, zorder=3))
        dagger |= flagged(d)
        texts.append(f"{labels[t]}{'†' if flagged(d) else ''} (mean mAUC {mean[i, m]:.1f} ± {sd[i, m]:.2f})")
        for j in xs:
            ax.annotate(f"{mean[i, j]:.1f}", (j, y[j]), xytext=(7 * side, 0), textcoords="offset points", ha="left" if side > 0 else "right", va="center",
                        fontsize=6, color=c, zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    for label, v in refs:  # reference value: dashed grey line across the Mean axis only
        y = (v - lo[m]) / span[m]
        ax.plot([m - 0.35, m + 0.5], [y, y], "--", color="0.5", lw=1, zorder=2)
        ax.text(m + 0.55, y, textwrap.fill(f"{label}: {v:g}", 22), fontsize=6.5, color="0.4", va="center")
    ax.set_xlim(-0.7, m + (2.6 if refs else 0.7))
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_frame_on(False)  # no box: only the vertical axes
    ax.legend(handles, texts, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1 + (len(handles) > 4), fontsize=8, frameon=False)
    runs = "/".join(sorted({str(d["runs"]) for d in got.values()}))
    fig.suptitle(f"{dataset_display(ds)}: frozen-backbone linear probe, {runs} runs, test mAUC" + (f"\n{title}" if title else ""), fontsize=12)
    cohort = f"\nn train/test: {first['n']['train']}/{first['n']['test']}"  # the page's tags share this n (checked above)
    fig.text(0.03, 0.02, FOOTER + (f"\n{DAGGER}" if dagger else "") + cohort, fontsize=7, color="0.3", va="bottom")
    return fig


def summary_rows(dss, tags, labels, res):
    """Header + one row of strings per tag: each dataset's mean mAUC (s.d.) in percent, then the plain mean of the four dataset means (blank if any is missing)."""
    rows = [["Backbone", *map(dataset_display, dss), "Mean of 4"]]
    for t in tags:
        ds_d = [res[ds].get(t) for ds in dss]
        cells = ["n/a" if d is None else f"{100 * d['test_mAUC_mean']:.1f} ({100 * d['test_mAUC_std']:.2f})" + ("†" if flagged(d) else "") for d in ds_d]
        full = len(dss) == len(DATASETS) and None not in ds_d
        mean4 = f"{np.mean([100 * d['test_mAUC_mean'] for d in ds_d]):.1f}" + ("†" if any(flagged(d) for d in ds_d) else "") if full else ""
        rows.append([labels[t], *cells, mean4])
    return rows


def summary_page(rows, title):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    n, h = len(rows[0]), min(0.1 * len(rows), 0.9)  # h: table height in axes fractions. Tag names are the first column, not row labels (matplotlib lets those hang off the page)
    w0 = min(0.012 * max(len(r[0]) for r in rows) + 0.03, 0.45)
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], colWidths=[w0] + [(1 - w0) / (n - 1)] * (n - 1), bbox=[0, 0.5 - h / 2, 1, h], cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    fig.suptitle("Summary: frozen-backbone linear probe, test mAUC % mean (s.d.) over runs" + (f"\n{title}" if title else ""), fontsize=12)
    dagger = any("†" in c for r in rows[1:] for c in r)
    fig.text(0.03, 0.02, "Mean of 4 = plain mean of the four dataset means; blank if a dataset is missing.\n" + FOOTER + (f"\n{DAGGER}" if dagger else ""), fontsize=7, color="0.3", va="bottom")
    return fig


def table_text(rows):
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(n) for c, n in zip(r, w)).rstrip() for r in rows)


def main(argv=None):
    a = parse_args(argv)
    if a.selftest:
        return selftest()
    root, dss = a.results_root, list(dict.fromkeys(a.datasets))
    if not os.path.isdir(root):
        sys.exit(f"plot_linear_probes: results_root {root!r} is not a directory")
    pairs = a.tags or [(t, t) for t in sorted(os.listdir(root)) if any(os.path.isfile(os.path.join(root, t, ds + ".json")) for ds in dss)]
    # ponytail: refused, not drawn twice (dict(pairs) gives one curve, tags gives two rows); key curves by position if it must be drawn
    dups = sorted({t for t, _ in pairs if [u for u, _ in pairs].count(t) > 1})
    if dups:
        note(f"--tags names a tag more than once: {', '.join(dups)}")
        sys.exit(2)
    labels, tags = dict(pairs), [t for t, _ in pairs]
    res = {ds: {} for ds in dss}
    for ds in dss:
        for t in tags:
            d = load(root, t, ds)
            if d:
                res[ds][t] = d
    if not any(res.values()):
        sys.exit(f"plot_linear_probes: no <tag>/<dataset>.json for tags {tags} under {root!r}")
    rows, figs = summary_rows(dss, tags, labels, res), []
    for ds in dss:  # all figures are built before the PDF is opened, so a refusal leaves no half-written file
        if res[ds]:
            figs.append(dataset_page(ds, {t: res[ds][t] for t in tags if t in res[ds]}, labels, [(l, v) for d, l, v in a.reference if d == ds], a.title, tags))
        else:
            note(f"no tag has {ds}: page skipped")
    figs.append(summary_page(rows, a.title))
    with PdfPages(a.out) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
    print(table_text(rows))
    note(f"wrote {a.out}")


def make_synthetic(root, tags=("rad", "stock", "stage"), classes=(6, 14, 14, 5), seed=0):
    """run_linearprobe.py-shaped JSONs (fractions, 10 runs) for every tag x dataset; the first tag is flagged for RAD-DINO overlap on ChestXray14."""
    rng = np.random.default_rng(seed)
    for k, tag in enumerate(tags):
        os.makedirs(os.path.join(root, tag), exist_ok=True)
        for ds, n in zip(DATASETS, classes):
            cls = [f"Class_{i}" for i in range(n)]
            auc = 0.70 + 0.03 * k + 0.10 * rng.random((10, n))  # run x class
            d = {"tag": tag, "dataset": ds, "model": "synthetic", "weights": {"path": "w.pth"}, "n": {"train": 100, "test": 50}, "max_images": None, "runs": 10, "classes": cls,
                 "per_run": [{"run": r, "seed": r, "best_epoch": 9, "test_mAUC": float(auc[r].mean()), "test_AUC": dict(zip(cls, auc[r].tolist()))} for r in range(10)],
                 "test_mAUC_mean": float(auc.mean(1).mean()), "test_mAUC_std": float(auc.mean(1).std()),
                 "test_AUC_mean": dict(zip(cls, auc.mean(0).tolist())), "test_AUC_std": dict(zip(cls, auc.std(0).tolist())),
                 "flags": {"selection": "best epoch on test (as shipped)", "rad_dino_pretraining_overlap": "train+val+test" if (k, ds) == (0, "ChestXray14") else None}}
            with open(os.path.join(root, tag, ds + ".json"), "w") as f:
                json.dump(d, f)


def n_pages(path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(path).pages)
    except ImportError:  # no pypdf: count the /Type /Page objects (not /Type /Pages)
        with open(path, "rb") as f:
            return len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", f.read()))


def selftest():
    labels = ("RAD-DINO", "DINOv3 stock", "DINO-X Cosine e30")
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic(tmp)
        pdf, buf = os.path.join(tmp, "out.pdf"), io.StringIO()
        with contextlib.redirect_stdout(buf):
            main([tmp, "--tags", *(f"{t}={l}" for t, l in zip(("rad", "stock", "stage"), labels)), "--out", pdf, "--title", "selftest",
                  "--reference", "VinDrCXR=Ark+ linear probe, Nature 2025 Fig. 3a=96.19"])
        print(buf.getvalue(), end="")
        assert n_pages(pdf) == 5, f"expected 5 pages (4 datasets + summary), got {n_pages(pdf)}"
        assert all(l in buf.getvalue() for l in labels), "summary is missing a label"
        assert "†" in buf.getvalue(), "overlap flag not marked in the summary"
        os.remove(os.path.join(tmp, "rad", "ChestXray14.json"))  # rad now lacks ChestXray14: that page must keep stock and stage on their full-order colours
        fig = dataset_page("ChestXray14", {t: load(tmp, t, "ChestXray14") for t in ("stock", "stage")}, dict(zip(("rad", "stock", "stage"), labels)), [], "", ["rad", "stock", "stage"])
        assert [c.lines[0].get_color() for c in fig.axes[0].containers] == [COLORS[1], COLORS[2]], "a tag's colour depends on which earlier tags its page has"
        plt.close(fig)
    # a smoke subset and a cohort mismatch are each refused: exit 1 through the CLI, the reason on stderr, no PDF written
    for tag, ds, change, msg in (("stock", "VinDrCXR", {"max_images": 512}, "stock/VinDrCXR: fitted on a smoke subset (max_images=512)"),
                                 ("stock", "CheXpert", {"n": {"train": 101, "test": 50}}, "CheXpert: tag stock has n {'train': 101, 'test': 50}")):
        with tempfile.TemporaryDirectory() as tmp:
            make_synthetic(tmp)
            path = os.path.join(tmp, tag, ds + ".json")
            with open(path) as f:
                d = json.load(f)
            with open(path, "w") as f:
                json.dump({**d, **change}, f)
            pdf = os.path.join(tmp, "out.pdf")
            r = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), tmp, "--tags", "rad", "stock", "stage", "--out", pdf], capture_output=True, text=True)
            assert r.returncode == 1 and msg in r.stderr and not os.path.exists(pdf), (r.returncode, r.stderr)
            print(r.stderr, end="")
    # a repeated --tags name is refused: exit 2 through the CLI, the reason on stderr, no PDF written
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic(tmp)
        pdf = os.path.join(tmp, "out.pdf")
        r = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), tmp, "--tags", "rad=A", "rad=B", "stock=C", "--out", pdf], capture_output=True, text=True)
        assert r.returncode == 2 and "plot_linear_probes: --tags names a tag more than once: rad" in r.stderr and not os.path.exists(pdf), (r.returncode, r.stderr)
        print(r.stderr, end="")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
