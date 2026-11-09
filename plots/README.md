# Plots

`auc_paths.py` makes `images/p31-auc-paths-00.svg` … `-06.svg`: the holdout AUC
path of 6 runs (3 per model), one more line on each slide.

```
python3 plots/auc_paths.py
```

Needs Python 3 and matplotlib (3.11.2 was used). The same data gives
byte-identical SVGs.

## Data

`data/` is a copy of the 20 files the plots need from
[szilard/xgboost-autoresearch-minimal3-runs](https://github.com/szilard/xgboost-autoresearch-minimal3-runs)
at commit `c34a0518a60288925065cb105d973771f5873c43` (2026-10-07), same paths as
in that repo:

- `run-multi/<group>/holdout_auc.tsv`: final holdout AUC of all 20 runs per
  model (used to choose the runs)
- for each plotted run: `holdout_scores.tsv`, `timing/runs.tsv`,
  `timing/clock.json`

To plot from a full copy of that repo instead: `python3 plots/auc_paths.py <path>`.

The paths are computed as in that repo's `tools/plot_holdout_auc.py`
(`holdout_auc_path_median.png`), with the same colours and y-range.

## Which runs, in which order

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
