"""Baselines for comparing the EBM results against.

An R-squared with nothing beside it says very little. These give the comparison
its meaning, in rising order of what they ask of the data:

1. `global_median`  Predict the median price for everything. By construction this
                    scores R-squared 0 on the target's own scale. It is here to
                    show what "no model" looks like in rupee terms.
2. `group_median`   Predict the median price of the same make/model group, falling
                    back to the global median for unseen levels. This is the real
                    bar, because it is what a person actually does - look up what
                    other cars of the same model are going for. A model that cannot
                    beat it is not earning its complexity.
3. `linear`         Ordinary least squares on one-hot encoded categoricals. On the
                    random split this is a **documented failure, not a usable
                    score** - and the cause is worth the space it takes.

                    `brand_grp` is nested inside `make_model_grp`: 156 of 158 model
                    levels map to exactly one brand. For any brand whose only
                    surviving model level is its own, the two dummy columns are
                    then *literally the same vector*. On the random split that
                    happens for Bajaj, whose single model is the Qute, so
                    `make_model_grp_Bajaj Qute` and `brand_grp_Bajaj` are
                    identical: the design matrix keeps one exact dependency even
                    after a reference level is dropped, and its condition number is
                    3.2e17. OLS then has no unique solution, and the minimum-norm
                    one `lstsq` returns carries coefficients large enough to
                    predict a negative power of ten in rupees on one rare row.

                    Note this is split-dependent rather than inevitable: on the
                    grouped split the same encoding comes out full rank with a
                    condition number of 825, and OLS is fine (R-squared 0.916).
                    The nesting is the mechanism; which rows land in training
                    decides whether it becomes exact. `conditioning_report()`
                    below measures it rather than asserting it.

                    The typical OLS prediction is still reasonable (median error
                    ~10%), which is why the median is reported beside the mean -
                    it is RMSE that a single wild prediction destroys.
4. `ridge`          The linear baseline **of record**, for exactly this reason.
                    Regularisation is not a refinement here; it is what makes a
                    linear fit dependable on this design matrix at all.
5. `random_forest`  A strong tabular baseline, and the thing EBM has to justify
                    itself against on accuracy.

All five were covered in lectures, which is why none of them can be the submitted
algorithm - and exactly why they make honest baselines.

The target is log10(price) throughout. Errors are reported in log10 units, which
is what the models are optimising, and also back-transformed to a percentage,
which is what a reader understands. Note that 10**prediction estimates the
*median* price for those features, not the mean - the back-transform is not
undone by exponentiating the mean of a log-scale fit.

The test set is not touched unless `--test` is passed. Hyperparameters are chosen
on validation; the test set is scored once, at the very end of the project.

Run: python code/train/baselines.py [--strategy random|grouped|both] [--test]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'processed' / 'vehicles.csv'
SPLITS = ROOT / 'data' / 'processed' / 'splits.csv'
OUT = ROOT / 'data' / 'processed' / 'baseline_scores.csv'

SEED = 42


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    sp = pd.read_csv(SPLITS)
    if len(sp) != len(df):
        raise SystemExit('splits.csv is stale - re-run split_data.py')
    df = df.merge(sp, on='id', validate='one_to_one')
    features.check(df.drop(columns=[c for c in sp.columns if c != 'id']))
    return df


def preprocessor(drop_reference: bool) -> ColumnTransformer:
    """Fitted on training rows only, inside the pipeline, so nothing leaks.

    `drop_reference` is not cosmetic. One-hot encoding every level of all nine
    categoricals alongside an intercept makes the design matrix rank-deficient by
    exactly nine columns - the dummy variable trap. Measured here: 258 columns,
    rank 249. Ordinary least squares then has infinitely many solutions, and the
    minimum-norm one `lstsq` returns carried coefficients up to 5.8e10, which
    extrapolated to a prediction of 10**-6.6e9 rupees on a single rare
    Bajaj/Unknown row and took validation R-squared to -2.1e17. Dropping one
    reference level per categorical removes the dependency and OLS behaves.

    Trees are the opposite case: collinearity is harmless to them, and dropping a
    level hides a category behind the absence of all the others, so the forest
    keeps the full encoding.
    """
    return ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), features.NUMERIC),
        # handle_unknown='ignore' matters: a model level present only in the test
        # split must not raise, it must simply contribute nothing.
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                              drop='first' if drop_reference else None),
         features.CATEGORICAL),
        ('flag', 'passthrough', features.FLAGS),
    ])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_* are log10 rupees. Percentage errors are back-transformed."""
    resid = y_pred - y_true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    pct = np.abs(10 ** y_pred - 10 ** y_true) / 10 ** y_true * 100
    return {
        'r2': 1 - ss_res / ss_tot,
        'rmse_log10': float(np.sqrt(np.mean(resid ** 2))),
        'mae_log10': float(np.mean(np.abs(resid))),
        'median_pct_err': float(np.median(pct)),
        'p90_pct_err': float(np.percentile(pct, 90)),
    }


def group_median_predict(train: pd.DataFrame, target: pd.DataFrame,
                         key: str = 'make_model_grp') -> np.ndarray:
    """What a person does by hand: look up what the same model sells for."""
    y = np.log10(train['price'])
    lookup = y.groupby(train[key]).median()
    fallback = float(y.median())
    return target[key].map(lookup).fillna(fallback).to_numpy(dtype=float)


# Whether each model needs a reference level dropped - see preprocessor().
DROP_REFERENCE = {'linear': True, 'ridge': True, 'random_forest': False}

MODELS = {
    'linear': lambda: LinearRegression(),
    'ridge': lambda: Ridge(alpha=10.0),
    'random_forest': lambda: RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.4,
        n_jobs=-1, random_state=SEED),
}


def run(df: pd.DataFrame, strategy: str, use_test: bool) -> list[dict]:
    col = f'split_{strategy}'
    train = df[df[col] == 'train']
    evals = [('val', df[df[col] == 'val'])]
    if use_test:
        evals.append(('test', df[df[col] == 'test']))

    print(f'\n{"=" * 78}\n{strategy.upper()} SPLIT  '
          f'(train {len(train):,} | ' +
          ' | '.join(f'{n} {len(d):,}' for n, d in evals) + f')\n{"=" * 78}')

    y_train = np.log10(train['price'].to_numpy())
    rows = []

    for name in ['global_median', 'group_median'] + list(MODELS):
        if name in MODELS:
            pipe = Pipeline([('prep', preprocessor(DROP_REFERENCE[name])),
                             ('model', MODELS[name]())])
            pipe.fit(train[features.FEATURES], y_train)
            predict = lambda d, _p=pipe: _p.predict(d[features.FEATURES])
        elif name == 'global_median':
            med = float(np.median(y_train))
            predict = lambda d, _m=med: np.full(len(d), _m)
        else:
            predict = lambda d: group_median_predict(train, d)

        for split_name, d in evals:
            m = metrics(np.log10(d['price'].to_numpy()), predict(d))
            rows.append({'strategy': strategy, 'split': split_name,
                         'model': name, **m})

    tbl = pd.DataFrame(rows)
    for split_name, _ in evals:
        sub = tbl[tbl['split'] == split_name].set_index('model')
        print(f'\n-- {split_name} --')
        print(sub[['r2', 'rmse_log10', 'mae_log10',
                   'median_pct_err', 'p90_pct_err']]
              .rename(columns={'r2': 'R2', 'rmse_log10': 'RMSE(log10)',
                               'mae_log10': 'MAE(log10)',
                               'median_pct_err': 'med %err',
                               'p90_pct_err': 'p90 %err'})
              .round({'R2': 4, 'RMSE(log10)': 4, 'MAE(log10)': 4,
                      'med %err': 1, 'p90 %err': 1}).to_string())
    return rows


def conditioning_report(df: pd.DataFrame, strategy: str) -> None:
    """Explain the OLS row, so a reader does not read it as a coding mistake."""
    train = df[df[f'split_{strategy}'] == 'train']
    print(f'\n-- why `linear` fails on the {strategy} split --')
    for drop in (False, True):
        X = preprocessor(drop).fit_transform(train[features.FEATURES])
        rank = np.linalg.matrix_rank(X)
        sv = np.linalg.svd(X, compute_uv=False)
        cond = sv[0] / max(sv[-1], 1e-300)
        print(f'  reference level dropped: {str(drop):<5} '
              f'columns {X.shape[1]:>3}  rank {rank:>3}  '
              f'deficiency {X.shape[1] - rank}  condition {cond:.2e}')
    # Name the offending columns rather than assert a general cause: whether the
    # nesting produces an *exact* dependency depends on which rows land in train.
    prep = preprocessor(True).fit(train[features.FEATURES])
    X = prep.transform(train[features.FEATURES])
    names = prep.get_feature_names_out()
    if np.linalg.matrix_rank(X) < X.shape[1]:
        null = np.linalg.svd(X, full_matrices=True)[2][-1]
        involved = [(float(null[i]), names[i])
                    for i in np.argsort(-np.abs(null))[:6] if abs(null[i]) > 1e-6]
        print('  exact dependency between:')
        for w, nm in involved:
            print(f'    {w:+.4f}  {nm}')
        nest = train.groupby('make_model_grp')['brand_grp'].nunique()
        print(f'  cause: brand is nested inside make_model ({(nest == 1).sum()} of '
              f'{len(nest)} model levels map to one brand), so for a brand whose')
        print('         only surviving model is its own, the two dummy columns '
              'are the same vector.')
        print('  consequence: OLS has no unique solution here. Ridge is the '
              'linear baseline of record.')
    else:
        print('  full rank on this split - OLS is well posed here. The failure is '
              'split-dependent,')
        print('  which is itself the argument for preferring the regularised fit.')


def band_breakdown(df: pd.DataFrame, strategy: str) -> None:
    """A single score hides a sixfold spread in relative error across the range."""
    col = f'split_{strategy}'
    train, val = df[df[col] == 'train'], df[df[col] == 'val']
    pipe = Pipeline([('prep', preprocessor(DROP_REFERENCE['random_forest'])),
                     ('model', MODELS['random_forest']())])
    pipe.fit(train[features.FEATURES], np.log10(train['price'].to_numpy()))
    pred = pipe.predict(val[features.FEATURES])
    truth = np.log10(val['price'].to_numpy())
    price = val['price'].to_numpy()

    print(f'\n-- random_forest on {strategy} validation, by price band --')
    bands = [(0, 2e6, '< Rs 2M'), (2e6, 5e6, 'Rs 2-5M'), (5e6, 1e7, 'Rs 5-10M'),
             (1e7, 2e7, 'Rs 10-20M'), (2e7, 5e7, 'Rs 20-50M'),
             (5e7, 1e10, '> Rs 50M')]
    for lo, hi, lbl in bands:
        msk = (price >= lo) & (price < hi)
        if msk.sum() < 10:
            continue
        pct = np.abs(10 ** pred[msk] - price[msk]) / price[msk] * 100
        print(f'  {lbl:<10} n={msk.sum():>4}  median %err {np.median(pct):>5.1f}  '
              f'RMSE(log10) {np.sqrt(np.mean((pred[msk] - truth[msk]) ** 2)):.4f}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', choices=['random', 'grouped', 'both'],
                    default='both')
    ap.add_argument('--test', action='store_true',
                    help='also score the held-out test set (use once, at the end)')
    args = ap.parse_args()

    df = load()
    strategies = ['random', 'grouped'] if args.strategy == 'both' else [args.strategy]

    rows: list[dict] = []
    for s in strategies:
        rows += run(df, s, args.test)
    conditioning_report(df, strategies[-1])
    band_breakdown(df, strategies[-1])

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nscores written to {OUT}')
    if not args.test:
        print('test set untouched - pass --test when the final model is settled')


if __name__ == '__main__':
    main()
