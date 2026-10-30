"""SHAP measured against the EBM's own decomposition.

Three things are produced.

1. **SHAP on the EBM against the EBM's exact decomposition.** For a GA2M the
   per-feature attribution is the feature's own shape function plus its share of
   every pairwise term it appears in. A pairwise term is a two-player game, so
   the symmetric split is half each. Features that appear in no interaction
   should therefore match SHAP exactly; features that do appear in one should
   differ only by however much the interaction is worth. Both are checked.

2. **SHAP on the random forest.** The contrast the report's argument rests on.
   The forest's explanation does not exist until SHAP constructs it, so this
   measures what that costs in time and whether it even agrees about which
   features matter.

3. **A local comparison** for the same vehicle as the report's Figure 8, so the
   two figures can be read side by side.

Why SHAP needs the inputs re-encoded
------------------------------------
`shap.maskers.Independent` validates its background data with `np.isfinite`,
which fails on the object columns the EBM is happy to take. The features are
therefore passed to SHAP as integer codes and decoded back inside the prediction
wrapper. `engine_cc` carries a sentinel for missing rather than NaN for the same
reason; it decodes back to NaN, so the background distribution still contains the
electric vehicles whose engine size was deliberately never imputed.

Run: python code/explain/shap_comparison.py [--samples N]
Output: data/processed/shap_comparison.csv, figures/fig14_shap_vs_ebm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features
import figstyle as fs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
from baselines import MODELS, preprocessor, DROP_REFERENCE   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / 'figures'
OUT = ROOT / 'data' / 'processed' / 'shap_comparison.csv'

ENGINE_MISSING = -1.0        # sentinel; engine_cc is otherwise >= 600
BACKGROUND = 100             # background rows for the marginal masker


# ------------------------------------------------------------------ encoding

class Codec:
    """Integer codes in, model-ready frame out.

    SHAP perturbs a numeric matrix; the EBM wants a frame with its original
    dtypes. This translates between the two and nothing else, so that the thing
    being explained is exactly the model being reported.
    """

    def __init__(self, train: pd.DataFrame):
        self.levels = {c: list(pd.unique(train[c].dropna()))
                       for c in features.CATEGORICAL}

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in features.FEATURES:
            if c in features.CATEGORICAL:
                index = {v: i for i, v in enumerate(self.levels[c])}
                # An unseen level becomes its own code, which the decoder turns
                # back into a string the model routes to its 'unseen' bin.
                cols.append(df[c].map(index).fillna(len(self.levels[c])).to_numpy(float))
            elif c == 'engine_cc':
                cols.append(df[c].fillna(ENGINE_MISSING).to_numpy(float))
            else:
                cols.append(df[c].to_numpy(float))
        return np.column_stack(cols)

    def decode(self, codes: np.ndarray) -> pd.DataFrame:
        codes = np.atleast_2d(codes)
        out = {}
        for j, c in enumerate(features.FEATURES):
            col = codes[:, j]
            if c in features.CATEGORICAL:
                lv = self.levels[c]
                out[c] = [lv[int(v)] if 0 <= int(v) < len(lv) else '__unseen__'
                          for v in col]
            elif c == 'engine_cc':
                out[c] = np.where(col == ENGINE_MISSING, np.nan, col)
            elif c in features.FLAGS:
                out[c] = col.astype(int)
            else:
                out[c] = col
        return pd.DataFrame(out)[features.FEATURES]


# -------------------------------------------------- the EBM's own attribution

def ebm_attribution(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature attribution read straight off the fitted EBM.

    `explain_local` returns one score per *term*, and five of the 35 terms are
    pairs. A pair is split evenly between its two features, which is the
    symmetric division a Shapley value of a two-player game makes.
    """
    data = model.explain_local(X)
    rows = []
    for i in range(len(X)):
        d = data.data(i)
        attrib = dict.fromkeys(features.FEATURES, 0.0)
        for name, score in zip(d['names'], d['scores']):
            if ' & ' in name:
                a, b = name.split(' & ')
                attrib[a] += score / 2
                attrib[b] += score / 2
            else:
                attrib[name] += score
        rows.append(attrib)
    return pd.DataFrame(rows)[features.FEATURES]


def interaction_share(model) -> pd.Series:
    """How much of each feature's attribution can come from a pairwise term.

    A feature in no interaction must agree with SHAP exactly; this is the budget
    for the ones that do.
    """
    share = dict.fromkeys(features.FEATURES, 0.0)
    for idx, name in enumerate(model.term_names_):
        if ' & ' not in name:
            continue
        weight = float(np.mean(np.abs(model.term_scores_[idx])))
        for f in name.split(' & '):
            share[f] += weight / 2
    return pd.Series(share)


# ------------------------------------------------------------- forest + SHAP

def forest_shap(train: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """TreeSHAP on the random forest, aggregated back to the 20 features.

    The forest sees a 258-column one-hot design matrix, so SHAP explains encoded
    columns. Summing a categorical's columns recovers the attribution for the
    feature itself, which is what can be compared with the EBM.
    """
    pipe = Pipeline([('prep', preprocessor(DROP_REFERENCE['random_forest'])),
                     ('model', MODELS['random_forest']())])
    pipe.fit(train[features.FEATURES], np.log10(train['price']))

    prep, forest = pipe.named_steps['prep'], pipe.named_steps['model']
    encoded = prep.transform(X)
    names = list(prep.get_feature_names_out())

    start = time.time()
    values = shap.TreeExplainer(forest).shap_values(encoded, check_additivity=True)
    elapsed = time.time() - start

    # num__age_years -> age_years, cat__brand_grp_Toyota -> brand_grp
    def owner(col: str) -> str:
        body = col.split('__', 1)[1]
        if body in features.FEATURES:
            return body
        for f in features.CATEGORICAL:
            if body.startswith(f + '_'):
                return f
        raise KeyError(col)

    out = pd.DataFrame(0.0, index=range(len(X)), columns=features.FEATURES)
    for j, col in enumerate(names):
        out[owner(col)] += values[:, j]
    return out, elapsed


# ------------------------------------------------------------------- figure

def figure(row_label: str, table: pd.DataFrame) -> None:
    """Three attributions for one vehicle, as price multipliers."""
    table = table.reindex(table['EBM, exact'].abs().sort_values().index)
    y = np.arange(len(table))
    height = 0.26

    fs.style()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for offset, (col, colour) in zip(
            (height, 0.0, -height),
            (('EBM, exact', fs.BLUE), ('SHAP on the EBM', fs.ORANGE),
             ('SHAP on the forest', fs.BASE))):
        ax.barh(y + offset, table[col], height=height, color=colour,
                linewidth=0, label=col)
    ax.set_yticks(y, table.index, fontsize=8.5)
    ax.axvline(0, color=fs.BASE, linewidth=0.8)
    ax.set_xlabel('Effect on price')
    span = float(table.abs().to_numpy().max()) * 1.3
    ax.set_xlim(-span, span)
    fs.multiplier_ticks(ax, 'x')
    fs.hgrid(ax)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    fs.title(fig, f'Three attributions for one vehicle: {row_label}',
             ax=ax)
    fs.save(fig, 'fig14_shap_vs_ebm', FIGS)


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=200,
                    help='test rows to explain (SHAP on the EBM is the slow part)')
    args = ap.parse_args()

    model = joblib.load(ROOT / 'models' / 'ebm_grouped.joblib')
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'vehicles.csv').merge(
        pd.read_csv(ROOT / 'data' / 'processed' / 'splits.csv'), on='id',
        validate='one_to_one')
    train = df[df['split_grouped'] == 'train']
    test = df[df['split_grouped'] == 'test']

    sample = test.sample(min(args.samples, len(test)), random_state=42)
    X = sample[features.FEATURES].reset_index(drop=True)

    codec = Codec(train)
    background = codec.encode(train[features.FEATURES].sample(BACKGROUND,
                                                              random_state=0))

    def predict(codes):
        return model.predict(codec.decode(codes))

    print(f'explaining {len(X)} test rows\n')

    start = time.time()
    explainer = shap.Explainer(predict, shap.maskers.Independent(background))
    shap_ebm = pd.DataFrame(explainer(codec.encode(X), silent=True).values,
                            columns=features.FEATURES)
    ebm_secs = time.time() - start

    exact = ebm_attribution(model, X)
    budget = interaction_share(model)
    in_pair = budget[budget > 0].index.tolist()
    solo = [f for f in features.FEATURES if f not in in_pair]

    diff = (shap_ebm - exact).abs()
    print('=' * 72)
    print('1. SHAP on the EBM vs the EBM read directly')
    print('=' * 72)
    print(f'  SHAP took {ebm_secs:.1f}s for {len(X)} rows '
          f'({ebm_secs / len(X) * 1000:.0f} ms each)')
    print(f'  reading the same numbers off the model: effectively free\n')
    print(f'  {len(solo)} features appear in no pairwise term:')
    print(f'    max absolute difference  {diff[solo].to_numpy().max():.2e}')
    print(f'    mean absolute difference {diff[solo].to_numpy().mean():.2e}')
    print(f'\n  {len(in_pair)} features do appear in one:')
    for f in sorted(in_pair, key=lambda c: -diff[c].mean()):
        print(f'    {f:<16} mean diff {diff[f].mean():.4f}   '
              f'interaction budget {budget[f]:.4f}')

    corr = float(pd.Series(shap_ebm.to_numpy().ravel()).corr(
        pd.Series(exact.to_numpy().ravel())))
    print(f'\n  correlation across all {diff.size} attributions: {corr:.6f}')

    print()
    print('=' * 72)
    print('2. SHAP on the random forest')
    print('=' * 72)
    shap_rf, rf_secs = forest_shap(train, X)
    print(f'  TreeSHAP took {rf_secs:.1f}s for {len(X)} rows '
          f'({rf_secs / len(X) * 1000:.0f} ms each)')

    importance = pd.DataFrame({
        'ebm_exact': exact.abs().mean(),
        'shap_ebm': shap_ebm.abs().mean(),
        'shap_forest': shap_rf.abs().mean(),
    })
    rank = importance.rank(ascending=False)
    print(f'\n  rank correlation, EBM vs forest: '
          f'{importance["ebm_exact"].corr(importance["shap_forest"], method="spearman"):.3f}')
    print('\n  mean |attribution| per feature:')
    print(importance.sort_values('ebm_exact', ascending=False)
          .head(10).to_string(float_format=lambda v: f'{v:.4f}'))

    importance.to_csv(OUT)
    print(f'\nwritten to {OUT}')

    # 3. the same vehicle as Figure 8, so the two can be compared directly
    val = df[df['split_grouped'] == 'val'].copy()
    val['pred'] = model.predict(val[features.FEATURES])
    val['err'] = (10 ** val['pred'] - val['price']).abs() / val['price']
    pick = val[(val['condition'] == 'Used') & val['price'].between(5e6, 12e6)]
    one = pick.loc[[(pick['err'] - pick['err'].median()).abs().idxmin()]]
    one_X = one[features.FEATURES].reset_index(drop=True)

    table = pd.DataFrame({
        'EBM, exact': ebm_attribution(model, one_X).iloc[0],
        'SHAP on the EBM': pd.Series(
            explainer(codec.encode(one_X), silent=True).values[0],
            index=features.FEATURES),
        'SHAP on the forest': forest_shap(train, one_X)[0].iloc[0],
    })
    table = table[table.abs().max(axis=1) > 0.005]
    figure(str(one['title'].iloc[0])[:44], table)
    print(f'\nfigure 14 drawn for: {one["title"].iloc[0]}')
    print(table.to_string(float_format=lambda v: f'{v:+.4f}'))


if __name__ == '__main__':
    main()
