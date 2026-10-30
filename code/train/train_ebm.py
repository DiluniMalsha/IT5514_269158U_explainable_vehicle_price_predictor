"""Fit the Explainable Boosting Machine

EBM is a Generalised Additive Model with pairwise interactions (GA2M). It fits one
shape function per feature by cyclic gradient boosting on shallow trees, then
optionally adds a bounded number of pairwise terms. The prediction is the sum of
those terms, so the model *is* its own explanation - there is no post-hoc
approximation standing between the reader and what the model actually does.

Tuning is deliberately narrow and done on the validation set only:

`interactions`   The question worth asking. Zero gives a purely additive model,
                 every term a single feature and trivially readable. More terms buy
                 accuracy and cost interpretability, and this dataset makes the
                 trade concrete - age and odometer correlate at 0.883, so there is
                 real joint structure for pairwise terms to pick up.
`learning_rate`  Standard, tuned second at the chosen interaction count.

Everything else is left at library defaults, including `outer_bags=14`. That
default matters for the report: EBM bags the whole fit, so each shape function
comes with a spread across bags, and the explainability figures can show
uncertainty instead of a single confident-looking line.

Missing values are passed through untouched. EBM gives them their own bin, which
is why `engine_cc` is left blank for electric vehicles rather than imputed - an
imputed engine size would invent a number for a car that has no engine.

The test set is untouched unless `--test` is passed. Use it once.

Run: python code/train/train_ebm.py [--quick] [--test]
Output: models/ebm_*.joblib, data/processed/ebm_scores.csv,
        data/processed/ebm_terms.csv
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'processed' / 'vehicles.csv'
SPLITS = ROOT / 'data' / 'processed' / 'splits.csv'
MODELS = ROOT / 'models'
SCORES = ROOT / 'data' / 'processed' / 'ebm_scores.csv'
TERMS = ROOT / 'data' / 'processed' / 'ebm_terms.csv'

SEED = 42

# Flags are 0/1, so nominal rather than continuous: a shape function over two
# points is a pair of offsets, and calling it continuous invites a reader to
# interpolate between them.
FEATURE_TYPES = ['nominal' if c in features.CATEGORICAL or c in features.FLAGS
                 else 'continuous' for c in features.FEATURES]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    sp = pd.read_csv(SPLITS)
    if len(sp) != len(df):
        raise SystemExit('splits.csv is stale - re-run split_data.py')
    return df.merge(sp, on='id', validate='one_to_one')


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


def fit(X, y, **kw) -> tuple[ExplainableBoostingRegressor, float]:
    t0 = time.time()
    m = ExplainableBoostingRegressor(feature_types=FEATURE_TYPES,
                                     random_state=SEED, **kw)
    m.fit(X, y)
    return m, time.time() - t0


def pick_elbow(rows: list[dict], keep_fraction: float = 0.10) -> int:
    """Choose the interaction count at the elbow of the accuracy/complexity curve.

    An absolute R-squared tolerance is the obvious rule and a bad one: the cutoff
    has to be picked by hand, and whatever is picked decides the answer, so the
    justification is circular. An earlier version of this function used a 0.002
    tolerance and selected 100 interactions purely because 40 missed the line by
    0.0007 - the threshold chose, not the data.

    What the data does show is where the *marginal* return collapses. R-squared
    gained per additional term, measured on this dataset:

        5 -> 15    +0.0102 over  10 terms = 0.00102 per term
        15 -> 40   +0.0023 over  25 terms = 0.00009 per term
        40 -> 100  +0.0027 over  60 terms = 0.00005 per term

    An eleven-fold drop after 15. That is an elbow in the data rather than a line
    drawn by the author, and it still works if someone re-runs this on new data.

    So: keep the largest configuration whose own marginal gain per term is at
    least `keep_fraction` of the best marginal gain seen. Anything past it is
    spending terms on rounding error.
    """
    ordered = sorted(rows, key=lambda r: r['interactions'])
    gains = []
    for prev, cur in zip(ordered, ordered[1:]):
        step = cur['terms'] - prev['terms']
        gains.append((cur, (cur['r2'] - prev['r2']) / step if step else 0.0))
    if not gains:
        return ordered[0]['interactions']

    best_gain = max(g for _, g in gains)
    worthwhile = [cfg for cfg, g in gains if g >= keep_fraction * best_gain]
    chosen = worthwhile[-1] if worthwhile else ordered[0]
    best = max(rows, key=lambda r: r['r2'])

    print('\n  marginal R2 per added term:')
    for cfg, g in gains:
        verdict = 'kept' if g >= keep_fraction * best_gain else 'not worth it'
        print(f'    interactions={cfg["interactions"]:<4} {g:>9.5f} per term   '
              f'{verdict}')
    print(f'\n  highest R2: interactions={best["interactions"]} '
          f'({best["r2"]:.4f}, {best["terms"]} terms)')
    print(f'  elbow:      interactions={chosen["interactions"]} '
          f'({chosen["r2"]:.4f}, {chosen["terms"]} terms)')
    print(f'  -> giving up {best["r2"] - chosen["r2"]:.4f} R2 to drop '
          f'{best["terms"] - chosen["terms"]} terms')
    return chosen['interactions']


def tune(df: pd.DataFrame, quick: bool, fixed: int | None = None,
         fixed_lr: float | None = None) -> dict:
    """Validation-set search. Narrow on purpose: two parameters, staged."""
    tr = df[df['split_grouped'] == 'train']
    va = df[df['split_grouped'] == 'val']
    X_tr, y_tr = tr[features.FEATURES], np.log10(tr['price'].to_numpy())
    X_va, y_va = va[features.FEATURES], np.log10(va['price'].to_numpy())

    print(f'tuning on the grouped split (train {len(tr):,} | val {len(va):,})')
    if fixed is None:
        print('stage 1: how many pairwise interaction terms are worth their '
              'opacity?\n')
        grid = [0, 5, 15] if quick else [0, 5, 15, 40, 100]
    else:
        print(f'stage 1 skipped - interactions fixed at {fixed}; the search it '
              f'came from is in data/processed/ebm_tuning.csv\n')
        grid = []
    rows = []
    for k in grid:
        m, secs = fit(X_tr, y_tr, interactions=k)
        s = metrics(y_va, m.predict(X_va))
        rows.append({'interactions': k, 'secs': secs, 'terms': len(m.term_names_), **s})
        print(f'  interactions={k:<4} {secs:>6.1f}s  terms {len(m.term_names_):>4}  '
              f'val R2 {s["r2"]:.4f}  med %err {s["median_pct_err"]:>5.1f}')

    chosen_k = fixed if fixed is not None else pick_elbow(rows)

    if fixed_lr is not None:
        print(f'stage 2 skipped - learning_rate fixed at {fixed_lr}; the search '
              f'it came from is in data/processed/ebm_tuning.csv')
        print(f'\nchosen: interactions={chosen_k}, learning_rate={fixed_lr}')
        return {'interactions': chosen_k, 'learning_rate': fixed_lr}

    print(f'\nstage 2: learning rate at interactions={chosen_k}\n')
    lr_rows = []
    for lr in ([0.015, 0.03] if quick else [0.008, 0.015, 0.03, 0.06]):
        m, secs = fit(X_tr, y_tr, interactions=chosen_k, learning_rate=lr)
        s = metrics(y_va, m.predict(X_va))
        lr_rows.append({'learning_rate': lr, 'secs': secs, **s})
        print(f'  learning_rate={lr:<6} {secs:>6.1f}s  val R2 {s["r2"]:.4f}  '
              f'med %err {s["median_pct_err"]:>5.1f}')

    best_lr = max(lr_rows, key=lambda r: r['r2'])['learning_rate']
    print(f'\nchosen: interactions={chosen_k}, learning_rate={best_lr}')
    out = ROOT / 'data' / 'processed' / 'ebm_tuning.csv'
    pd.DataFrame(rows + lr_rows).to_csv(
        out, index=False, mode='a' if fixed is not None else 'w',
        header=fixed is None)
    return {'interactions': chosen_k, 'learning_rate': best_lr}


def final(df: pd.DataFrame, params: dict, use_test: bool) -> list[dict]:
    rows = []
    for strategy in ('grouped', 'random'):
        col = f'split_{strategy}'
        tr = df[df[col] == 'train']
        X_tr, y_tr = tr[features.FEATURES], np.log10(tr['price'].to_numpy())
        m, secs = fit(X_tr, y_tr, **params)

        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, MODELS / f'ebm_{strategy}.joblib')

        splits = ['train', 'val'] + (['test'] if use_test else [])
        print(f'\n-- EBM on the {strategy} split ({secs:.1f}s, '
              f'{len(m.term_names_)} terms) --')
        for name in splits:
            d = df[df[col] == name]
            s = metrics(np.log10(d['price'].to_numpy()),
                        m.predict(d[features.FEATURES]))
            rows.append({'strategy': strategy, 'split': name, 'model': 'ebm',
                         **params, **s})
            print(f'  {name:<6} n={len(d):>5}  R2 {s["r2"]:.4f}  '
                  f'RMSE {s["rmse_log10"]:.4f}  med %err {s["median_pct_err"]:>5.1f}  '
                  f'p90 %err {s["p90_pct_err"]:>5.1f}')

        if strategy == 'grouped':
            g = m.explain_global().data()
            pd.DataFrame({'term': g['names'], 'importance': g['scores']}) \
              .sort_values('importance', ascending=False) \
              .to_csv(TERMS, index=False)
            print('\n  top terms by importance:')
            for n, sc in sorted(zip(g['names'], g['scores']),
                                key=lambda t: -t[1])[:12]:
                print(f'    {n:<40} {sc:.4f}')

            # Per-band error: a single score hides a large spread.
            va = df[df[col] == 'val']
            pred = m.predict(va[features.FEATURES])
            price = va['price'].to_numpy()
            print('\n  validation error by price band:')
            for lo, hi, lbl in [(0, 2e6, '< Rs 2M'), (2e6, 5e6, 'Rs 2-5M'),
                                (5e6, 1e7, 'Rs 5-10M'), (1e7, 2e7, 'Rs 10-20M'),
                                (2e7, 5e7, 'Rs 20-50M'), (5e7, 1e10, '> Rs 50M')]:
                msk = (price >= lo) & (price < hi)
                if msk.sum() < 10:
                    continue
                pct = np.abs(10 ** pred[msk] - price[msk]) / price[msk] * 100
                print(f'    {lbl:<10} n={msk.sum():>4}  '
                      f'median %err {np.median(pct):>5.1f}')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='smaller search grid')
    ap.add_argument('--interactions', type=int, default=None,
                    help='skip the stage-1 search and fix the interaction count')
    ap.add_argument('--learning-rate', type=float, default=None,
                    dest='learning_rate',
                    help='skip the stage-2 search and fix the learning rate')
    ap.add_argument('--test', action='store_true',
                    help='also score the held-out test set (use once)')
    args = ap.parse_args()

    df = load()
    params = tune(df, args.quick, args.interactions, args.learning_rate)
    rows = final(df, params, args.test)

    pd.DataFrame(rows).to_csv(SCORES, index=False)
    (ROOT / 'models' / 'ebm_params.json').write_text(json.dumps(params, indent=2))
    print(f'\nscores -> {SCORES}')
    print(f'models -> {MODELS}')
    if not args.test:
        print('test set untouched - pass --test when the model is settled')


if __name__ == '__main__':
    main()
