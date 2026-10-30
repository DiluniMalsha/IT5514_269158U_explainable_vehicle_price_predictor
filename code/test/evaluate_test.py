"""The single scoring of the held-out test set.

Run once, when the model is final. Everything before this point was chosen on the
validation set; the test set exists to give one honest number at the end, and it
stops being that the moment a choice is made in response to it.

This loads the saved models rather than refitting. The fit is deterministic
(`random_state=42`, fixed split file), so the numbers are identical to those from
`train_ebm.py --interactions 15 --learning-rate 0.06 --test`, which remains the
canonical reproduction command.

Run: python code/test/evaluate_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'processed' / 'test_scores.csv'

BANDS = [(0, 2e6, '< Rs 2M'), (2e6, 5e6, 'Rs 2-5M'), (5e6, 1e7, 'Rs 5-10M'),
         (1e7, 2e7, 'Rs 10-20M'), (2e7, 5e7, 'Rs 20-50M'), (5e7, 1e10, '> Rs 50M')]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_pred - y_true
    pct = np.abs(10 ** y_pred - 10 ** y_true) / 10 ** y_true * 100
    return {
        'r2': float(1 - np.sum(resid ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        'rmse_log10': float(np.sqrt(np.mean(resid ** 2))),
        'mae_log10': float(np.mean(np.abs(resid))),
        'median_pct_err': float(np.median(pct)),
        'p90_pct_err': float(np.percentile(pct, 90)),
    }


def main() -> None:
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'vehicles.csv').merge(
        pd.read_csv(ROOT / 'data' / 'processed' / 'splits.csv'), on='id',
        validate='one_to_one')

    rows = []
    for strategy in ('grouped', 'random'):
        m = joblib.load(ROOT / 'models' / f'ebm_{strategy}.joblib')
        col = f'split_{strategy}'
        print(f'\n{"=" * 70}\nEBM on the {strategy} split ({len(m.term_names_)} terms)'
              f'\n{"=" * 70}')
        for split in ('train', 'val', 'test'):
            d = df[df[col] == split]
            s = metrics(np.log10(d['price'].to_numpy()),
                        m.predict(d[features.FEATURES]))
            rows.append({'strategy': strategy, 'split': split, 'model': 'ebm', **s})
            mark = '  <- the one that counts' if split == 'test' else ''
            print(f'  {split:<6} n={len(d):>5}  R2 {s["r2"]:.4f}  '
                  f'RMSE {s["rmse_log10"]:.4f}  med %err {s["median_pct_err"]:>5.1f}  '
                  f'p90 %err {s["p90_pct_err"]:>5.1f}{mark}')

        if strategy == 'grouped':
            te = df[df[col] == 'test']
            pred = m.predict(te[features.FEATURES])
            price = te['price'].to_numpy()
            print('\n  test error by price band:')
            for lo, hi, lbl in BANDS:
                msk = (price >= lo) & (price < hi)
                if msk.sum() < 10:
                    continue
                pct = np.abs(10 ** pred[msk] - price[msk]) / price[msk] * 100
                print(f'    {lbl:<10} n={msk.sum():>4}  '
                      f'median %err {np.median(pct):>5.1f}')

            print('\n  test error by condition:')
            for cond in ('Brand New', 'Import', 'Reconditioned', 'Used'):
                msk = te['condition'].to_numpy() == cond
                if msk.sum() < 10:
                    continue
                pct = np.abs(10 ** pred[msk] - price[msk]) / price[msk] * 100
                print(f'    {cond:<14} n={msk.sum():>4}  '
                      f'median %err {np.median(pct):>5.1f}')

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
