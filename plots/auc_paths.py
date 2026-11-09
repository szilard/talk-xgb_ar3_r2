#!/usr/bin/env python3
"""Frames for the slides: holdout AUC paths appearing one run at a time.

Reads the runs from plots/data/, a copy of the files needed from
szilard/xgboost-autoresearch-minimal3-runs at commit c34a051 (see plots/README.md),
or from another copy of that repo given as the argument. Per run only
holdout_scores.tsv, timing/runs.tsv and timing/clock.json are needed. Writes
images/p31-auc-paths-00.svg (empty axes) ... -NN.svg (all runs).

A path is computed exactly as in that repo's tools/plot_holdout_auc.py: x = minutes
from the run's clock start to the end of each kept commit's harness run, y = its
holdout AUC, drawn as steps until the run's clock stopped. Colours and style follow
that script too. All frames have the same size and axes, so stepping through the
slides looks like lines being added. The SVGs are reproducible: the same data gives
byte-identical files.

Usage: plots/auc_paths.py [path to the runs repo]   (default: plots/data)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (group, model name, colour, runs in the order they appear); the runs are shown
# alternating between the models, the first model first
MODELS = [("sol6_n20", "gpt-6-sol", "#eb6834", [3, 10, 5]),
          ("astra6_n20", "gpt-6-astra", "#2a78d6", [9, 6, 11])]
YLIM = (0.6715, 0.6915)  # same range as run-multi/SUMMARY/holdout_auc_path_median.png
XLIM = (0, 62)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "images"

plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "auc-paths", "font.size": 13})


def keep_path(run_dir):
    clock = json.loads((run_dir / "timing" / "clock.json").read_text())
    end = {}
    with open(run_dir / "timing" / "runs.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] == "ok":
                end.setdefault(r["commit"], (float(r["end"]) - clock["start"]) / 60)
    path = []
    with open(run_dir / "holdout_scores.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"] == "keep" and r["holdout_auc"] not in ("", "N/A", "CRASH"):
                path.append((end[r["commit"]], float(r["holdout_auc"])))
    t, auc = zip(*path)
    return list(t) + [(clock["stop"] - clock["start"]) / 60], list(auc) + [auc[-1]]


def main(repo):
    order = [(group, colour, k) for runs in zip(*([(g, c, k) for k in ks] for g, _, c, ks in MODELS))
             for group, colour, k in runs]
    paths = []
    for group, colour, k in order:
        run_dir = Path(repo) / "run-multi" / group / f"{group}-{k}"
        paths.append((keep_path(run_dir), colour))

    for n in range(len(paths) + 1):
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=SURFACE)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.2)
        for (t, auc), colour in paths[:n]:
            ax.step(t, auc, where="post", color=colour, lw=2.2, zorder=2)
            ax.plot(t[-1], auc[-1], "o", color=colour, ms=8, zorder=3)  # dot at the end: the final AUC
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.set_xlabel("minutes since the clock started", color=INK2, fontsize=17)
        ax.set_ylabel("holdout AUC", color=INK2, fontsize=17)
        legend = [Line2D([], [], color=c, lw=2.6, label=m) for _, m, c, _ in MODELS]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend),
                   frameon=False, labelcolor=INK2, fontsize=17, handlelength=2.5, handletextpad=0.6, columnspacing=2)
        out = OUT / f"p31-auc-paths-{n:02d}.svg"
        fig.savefig(out, facecolor=SURFACE, metadata={"Date": None})
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    main(sys.argv[1] if len(sys.argv) == 2 else DATA)
