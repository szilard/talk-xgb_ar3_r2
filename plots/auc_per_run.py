#!/usr/bin/env python3
"""Frames for the slides: final holdout AUC per run, one row per model, built up in steps.

Reads each model's run-multi/<group>/holdout_auc.tsv from plots/data/ (a copy from
szilard/xgboost-autoresearch-minimal3-runs at commit c34a051, see plots/README.md),
or from another copy of that repo given as the argument. Writes
images/p33-auc-per-run-1.svg ... -3.svg:

  1. astra and sol: a dot per run, the full range, the 10th/90th percentiles
  2. the same plus luna
  3. the same plus each model's mean with its 95% confidence interval

The plot is the beeswarm of that repo's tools/plot_holdout_auc.py
(holdout_auc_beeswarm.png), drawn the same way, with larger fonts and dots for the
slides. All frames have the same size, axes and rows, so stepping through the slides
looks like things being added. The SVGs are reproducible: the same data gives
byte-identical files. Runs with valid = "no" would be left out, as there; caveat runs
are drawn like the others (their caveats are all of a kind that repo draws plainly).

Usage: plots/auc_per_run.py [path to the runs repo]   (default: plots/data)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# (group, model name, colour); rows top to bottom
MODELS = [("astra6_n20", "gpt-6-astra", "#2a78d6"),
          ("sol6_n20", "gpt-6-sol", "#eb6834"),
          ("luna6_n20", "gpt-6-luna", "#1baf7a")]
# what each frame shows: the models with runs, and whether the means are drawn
FRAMES = [(["gpt-6-astra", "gpt-6-sol"], False),
          (["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"], False),
          (["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"], True)]
# t quantile 0.975 with n - 1 = 19 degrees of freedom (scipy.stats.t.ppf(0.975, 19)); every
# model has 20 runs. Kept here so the script needs no scipy; checked in main().
T975 = {19: 2.093024054408263}
SURFACE, INK, INK2, GRID, RANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
DOT = 120  # marker area, pt^2
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "images"

plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "auc-per-run", "font.size": 15})


def load(repo, group):
    with open(Path(repo) / "run-multi" / group / "holdout_auc.tsv") as f:
        return np.array([float(r["holdout_auc"]) for r in csv.DictReader(f, delimiter="\t")
                         if r["valid"] in ("yes", "caveat") and r["holdout_auc"]])


def swarm_offsets(xs, ax, y, diameter_pt):
    """Vertical offsets (data units) so that dots at xs on row y don't overlap (as in that repo).

    Dots are placed left to right; each takes the first free slot of 0, -1, -2, ... dot
    diameters below the row's centre line, so a dot only leaves the line when it would touch another.
    """
    d = diameter_pt * ax.figure.dpi / 72
    px = ax.transData.transform(np.column_stack([xs, np.full(len(xs), y)]))[:, 0]
    dy_per_px = 1 / (ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    placed, offsets = [], np.zeros(len(xs))
    for i in np.argsort(px):
        for k in range(len(xs)):
            slot = -k * d
            if all((px[i] - pj) ** 2 + (slot - sj) ** 2 >= d ** 2 for pj, sj in placed):
                break
        placed.append((px[i], slot))
        offsets[i] = slot * dy_per_px
    return offsets


def main(repo):
    runs = {m: load(repo, g) for g, m, _ in MODELS}
    for m, x in runs.items():
        if len(x) - 1 not in T975:
            sys.exit(f"{m} has {len(x)} runs; add the t quantile for {len(x) - 1} df to T975")
    lo = min(x.min() for x in runs.values())
    hi = max(x.max() for x in runs.values())
    pad = (hi - lo) * 0.05

    for n, (shown, means) in enumerate(FRAMES, start=1):
        fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.25)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.75, len(MODELS) - 0.55)  # before the swarm: it needs the final data -> pixel scale
        for i, (_, m, colour) in enumerate(MODELS):
            if m not in shown:
                continue
            y = len(MODELS) - 1 - i
            x = runs[m]
            ax.plot([x.min(), x.max()], [y, y], color=RANGE, lw=9, solid_capstyle="round", zorder=1)
            dys = swarm_offsets(x, ax, y, np.sqrt(DOT) + 1.4 + 0.8)
            ax.scatter(x, y + dys, s=DOT, zorder=3, linewidths=1.4, facecolors=colour, edgecolors=colour)
            # "nearest": each percentile is an actual run
            p10, p90 = np.percentile(x, [10, 90], method="nearest")
            ax.scatter([p10, p90], [y, y], marker="|", s=520, color=INK, linewidths=2.2, zorder=4)
            if means:
                half = T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
                ax.errorbar(x.mean(), y + 0.25, xerr=half, fmt="none", ecolor=INK, elinewidth=2.2, capsize=5, zorder=4)
                ax.scatter(x.mean(), y + 0.25, s=70, color=INK, zorder=5)

        ax.set_yticks(range(len(MODELS)))
        labels = ax.set_yticklabels([m for _, m, _ in reversed(MODELS)], color=INK, fontsize=17)
        for label in labels:  # a row's label only once it has runs; hidden, so the layout stays put
            label.set_alpha(1 if label.get_text() in shown else 0)
        ax.set_xlabel("holdout AUC", color=INK2, fontsize=17)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.tick_params(axis="y", pad=10)

        legend = [
            Line2D([], [], marker="o", ls="", color=INK2, markersize=10, label="run"),
            Line2D([], [], color=RANGE, lw=9, solid_capstyle="butt", label="full range"),
            Line2D([], [], marker="|", ls="", color=INK2, markersize=14, markeredgewidth=2.2,
                   label="10th and 90th percentile"),
            Line2D([], [], marker="o", color=INK, markersize=7, lw=2.2, label="mean, 95% CI of the mean"),
        ]
        leg = fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend),
                         frameon=False, fontsize=15, labelcolor=INK2, handletextpad=0.5, columnspacing=1.8)
        if not means:  # the mean's entry hidden, not left out, so the legend doesn't move
            leg.legend_handles[-1].set_alpha(0)
            leg.get_texts()[-1].set_alpha(0)

        out = OUT / f"p33-auc-per-run-{n}.svg"
        fig.savefig(out, facecolor=SURFACE, metadata={"Date": None})
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    main(sys.argv[1] if len(sys.argv) == 2 else DATA)
