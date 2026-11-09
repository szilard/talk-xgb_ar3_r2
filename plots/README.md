# Plots

- `auc_paths.py` makes `images/p31-auc-paths-00.svg` … `-06.svg`: the holdout
  AUC path of 6 runs (3 per model), one more line on each slide.
- `auc_per_run.py` makes `images/p33-auc-per-run-1.svg` … `-3.svg`: the final
  holdout AUC of all 20 runs per model, one row per model (the repo's
  `holdout_auc_beeswarm.png`), built up as 1. astra + sol, 2. + luna,
  3. + each model's mean and 95% CI.

```
python3 plots/auc_paths.py
python3 plots/auc_per_run.py
```

Needs Python 3 and matplotlib (3.11.2 was used); no scipy (the one t quantile
needed is in the script). The same data gives byte-identical SVGs.

## Data

`data/` is a copy of the 21 files the plots need from
[szilard/xgboost-autoresearch-minimal3-runs](https://github.com/szilard/xgboost-autoresearch-minimal3-runs)
at commit `c34a0518a60288925065cb105d973771f5873c43` (2026-10-07), same paths as
in that repo:

- `run-multi/<group>/holdout_auc.tsv` for sol, astra and luna: final holdout
  AUC of all 20 runs per model (`auc_per_run.py`; for sol and astra also used to
  choose the runs of `auc_paths.py`)
- for each plotted run: `holdout_scores.tsv`, `timing/runs.tsv`,
  `timing/clock.json`

To plot from a full copy of that repo instead: `python3 plots/<script> <path>`.

Both follow that repo's `tools/plot_holdout_auc.py` (`holdout_auc_path_median.png`,
`holdout_auc_beeswarm.png`): same computation and colours, larger fonts for the
slides.

## Which runs, in which order (`auc_paths.py`)

Shown alternating, sol first: **sol-3, astra-9, sol-10, astra-6, sol-5, astra-11**
(set in `MODELS` at the top of the script).

- Each trio is close to its model's 20 runs (final holdout AUC):

  | model | 20 runs: mean / sd | trio | trio: mean / sd |
  |---|---|---|---|
  | gpt-6-sol | 0.68418 / 0.00184 | 3, 10, 5 | 0.68437 / 0.00191 |
  | gpt-6-astra | 0.68656 / 0.00216 | 9, 6, 11 | 0.68647 / 0.00214 |

- Final AUCs in the order A1 > S2 > A3 > A2 > S1 > S3 (A/S = astra/sol, numbered
  in the order shown): 0.6888 > 0.6864 > 0.6860 > 0.6846 > 0.6841 > 0.6826, so the
  ranking is A S A A S S: astra ahead in 7 of the 9 astra-sol pairs, not all.
- sol-3 for S1 because it climbs in steps over the first 15 minutes (sol-2 jumps
  up in the first 5, sol-7 late, around minute 37).
- All 6 runs are valid without a caveat.
