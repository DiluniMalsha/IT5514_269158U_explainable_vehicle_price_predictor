"""The explainability figures

An EBM's prediction is the sum of its terms, so each shape function below is literally a
slice of the fitted model rather than a post-hoc approximation of one. That
distinction is the whole argument for choosing EBM, and these figures are where
the report makes it visible.

Reading the y-axes
------------------
The model is fitted on log10(price), so every term's contribution *adds* in log
space and therefore *multiplies* in rupees. The axes are relabelled as multipliers
(x0.5, x2) because that is how a reader thinks, and because it is the honest unit:
a car does not lose "Rs 2 million" for being eight years old, it loses a
percentage. The intercept is the base the multipliers apply to.

Uncertainty
-----------
EBM bags its fit (`outer_bags=14` by default), so each bin has a spread across
bags. Every shape function is drawn with a +/-1 standard-deviation band. The band
widens where the data thins out - that is the figure telling you which parts of
the curve to trust, and it is why no extra "data density" annotation is needed.

Missing values
--------------
EBM's own visualisations hide the missing-value bin, and it warns about this. Here
the missing bin is reported explicitly, because for `engine_cc` it is not an
absence of information - it is the electric vehicles, whose engine size was
deliberately blanked during cleaning rather than imputed.

Run: python code/explain/explain_figures.py
Output: figures/fig9..fig13
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features
import figstyle as fs

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / 'figures'
MODEL = ROOT / 'models' / 'ebm_grouped.joblib'


# --- pulling shape functions out of the fitted model ---------------------------

def term_index(m, name: str) -> int:
    return m.term_names_.index(name)


def continuous_shape(m, name: str):
    """Edges, per-bin score and spread for a single continuous term.

    `term_scores_` is laid out as [missing, bin_1 .. bin_n, unseen]; the first and
    last slots are not part of the curve and are returned separately.
    """
    i = term_index(m, name)
    f = m.term_features_[i][0]
    cuts = np.asarray(m.bins_[f][0], dtype=float)
    scores = np.ravel(m.term_scores_[i]).astype(float)
    sds = np.ravel(m.standard_deviations_[i]).astype(float)
    lo, hi = (float(v) for v in m.feature_bounds_[f])
    edges = np.concatenate([[lo], cuts, [hi]])
    body, sd = scores[1:-1], sds[1:-1]
    if len(body) != len(edges) - 1:          # layout differs from expectation
        n = min(len(body), len(edges) - 1)
        body, sd, edges = body[:n], sd[:n], edges[:n + 1]
    return edges, body, sd, float(scores[0]), float(sds[0])


def nominal_effects(m, name: str) -> pd.DataFrame:
    """Per-level score and spread for a categorical term."""
    i = term_index(m, name)
    f = m.term_features_[i][0]
    mapping = m.bins_[f][0]
    scores = np.ravel(m.term_scores_[i]).astype(float)
    sds = np.ravel(m.standard_deviations_[i]).astype(float)
    rows = [{'level': lvl, 'score': scores[c], 'sd': sds[c]}
            for lvl, c in mapping.items() if c < len(scores)]
    return pd.DataFrame(rows).sort_values('score')


def pair_grid(m, name: str):
    """2-D scores for a pairwise term, with the bin edges used for interactions."""
    i = term_index(m, name)
    fa, fb = m.term_features_[i]
    grid = np.asarray(m.term_scores_[i], dtype=float)

    def axis_edges(f, n_bins):
        res = m.bins_[f]
        cuts = np.asarray(res[min(len(res) - 1, 1)], dtype=float)
        lo, hi = (float(v) for v in m.feature_bounds_[f])
        edges = np.concatenate([[lo], cuts, [hi]])
        return edges[:n_bins + 1]

    inner = grid[1:-1, 1:-1]
    return axis_edges(fa, inner.shape[0]), axis_edges(fb, inner.shape[1]), inner


# --- figures -------------------------------------------------------------------

def fig_importance(m, cap: dict) -> None:
    """What the model leans on. Two categories of term, so two colours."""
    g = m.explain_global().data()
    t = pd.DataFrame({'term': g['names'], 'importance': g['scores']})
    t['pairwise'] = t['term'].str.contains(' & ')
    t = t.sort_values('importance', ascending=False).head(18)[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = [fs.ORANGE if p else fs.BLUE for p in t['pairwise']]
    ax.barh(np.arange(len(t)), t['importance'], color=colors, height=0.68,
            linewidth=0)
    ax.set_yticks(np.arange(len(t)), t['term'], fontsize=8.5)
    ax.set_xlabel('Mean absolute contribution to the prediction (log$_{10}$ price)')
    ax.set_xlim(0, t['importance'].max() * 1.12)
    fs.hgrid(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=fs.BLUE),
               plt.Rectangle((0, 0), 1, 1, color=fs.ORANGE)]
    ax.legend(handles, ['single feature', 'pairwise interaction'],
              loc='lower right')

    n_pair = int(t['pairwise'].sum())
    fs.title(fig, 'Age and engine size dominate; interactions are '
                  'secondary', ax=ax)
    fs.save(fig, 'fig9_term_importance', FIGS)

    main = t[~t['pairwise']].nlargest(2, 'importance')
    cap['f9'] = (f'top terms {", ".join(main["term"])}; '
                 f'{n_pair} pairwise terms in the top 18')


def fig_numeric_shapes(m, cap: dict) -> None:
    """The centrepiece: what the model learned about age, odometer and engine."""
    specs = [('age_years', 'Vehicle age (years)', 0, 35, None),
             ('mileage_km', 'Odometer (km)', 0, 300_000, 1000),
             ('engine_cc', 'Engine capacity (cc)', 500, 4000, None)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2))

    notes: dict = {}
    missing_bins: dict = {}
    for ax, (name, xlabel, xlo, xhi, div) in zip(axes, specs):
        edges, body, sd, miss, miss_sd = continuous_shape(m, name)
        x = edges / div if div else edges
        lo, hi = (xlo / div, xhi / div) if div else (xlo, xhi)

        ax.stairs(body, x, color=fs.BLUE, linewidth=2, baseline=None)
        ax.stairs(body + sd, x, baseline=body - sd, fill=True,
                  color=fs.BLUE, alpha=0.18, linewidth=0)
        ax.axhline(0, color=fs.BASE, linewidth=0.8)

        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel + (' (thousands)' if div else ''))
        inside = (edges[:-1] >= xlo) & (edges[:-1] <= xhi)
        pad = 0.12
        ax.set_ylim(min(body[inside]) - pad, max(body[inside]) + pad)
        fs.multiplier_axis(ax, *ax.get_ylim())
        fs.vgrid(ax)
        ax.set_title(name)
        notes[name] = (float(min(body[inside])), float(max(body[inside])))

        # The missing-value bin, which EBM's own plots leave out. Only drawn when
        # it falls inside the visible range: mileage_km's missing bin sits at
        # x0.36 from just 4 rows, and annotating it off-scale stretched the whole
        # figure with a stray label below the axes.
        ylo, yhi = ax.get_ylim()
        if abs(miss) > 1e-9 and ylo < miss < yhi:
            ax.axhline(miss, color=fs.ORANGE, linewidth=1.8)
            ax.annotate(f'not recorded: x{10 ** miss:.2f}',
                        xy=(0.96, miss), xycoords=('axes fraction', 'data'),
                        ha='right', va='bottom', fontsize=8, color=fs.ORANGE)
        missing_bins[name] = (miss, miss_sd)

    axes[0].set_ylabel('Effect on price')
    fs.title(fig, 'Fitted shape functions, with the spread across '
                  'bagged fits', y=1.06, ax=axes[0])
    fs.save(fig, 'fig10_shape_functions', FIGS)

    # Reported in full here, including the bins too far off-scale to draw.
    print('\n  missing-value bins (EBM hides these in its own plots):')
    for name, (sc, sd) in missing_bins.items():
        n_missing = int(pd.isna(pd.read_csv(
            ROOT / 'data' / 'processed' / 'vehicles.csv')[name]).sum())
        if abs(sc) < 1e-9:
            print(f'    {name:<12} no missing values')
        else:
            print(f'    {name:<12} x{10 ** sc:.2f} (sd {sd:.3f}) '
                  f'from {n_missing} rows')

    cap['f10'] = (f'age spans x{10 ** notes["age_years"][0]:.2f}'
                  f'-x{10 ** notes["age_years"][1]:.2f}; '
                  f'engine_cc missing bin '
                  f'x{10 ** missing_bins["engine_cc"][0]:.2f}')


def fig_categoricals(m, cap: dict) -> None:
    """The small categoricals, where every level has enough rows to be read."""
    names = ['condition', 'fuel_type', 'transmission', 'body']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))

    for ax, name in zip(axes.ravel(), names):
        t = nominal_effects(m, name)
        t = t[t['level'].astype(str) != 'Unknown']
        ax.barh(np.arange(len(t)), t['score'], xerr=t['sd'],
                color=fs.BLUE, height=0.64, linewidth=0,
                error_kw=dict(ecolor=fs.BASE, elinewidth=1, capsize=0))
        ax.set_yticks(np.arange(len(t)), t['level'].astype(str), fontsize=8.5)
        ax.axvline(0, color=fs.BASE, linewidth=0.8)
        ax.set_title(f'{name}   (spread x{10 ** t["score"].min():.2f} '
                     f'to x{10 ** t["score"].max():.2f})', fontsize=9.5)
        fs.hgrid(ax)
        span = max(abs(t['score'])) * 1.35 + 0.02
        ax.set_xlim(-span, span)
        fs.multiplier_ticks(ax, 'x')

    # tight_layout before the title: without it the lower row's panel titles
    # print straight on top of the upper row's tick labels.
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    fs.title(fig, 'Categorical effects, as price multipliers',
             y=1.03, ax=axes[0][0])
    fs.save(fig, 'fig11_categorical_effects', FIGS)

    cond = nominal_effects(m, 'condition')
    cap['f11'] = '; '.join(f'{r.level} x{10 ** r.score:.2f}'
                           for r in cond.itertuples() if str(r.level) != 'Unknown')


def fig_interactions(m, cap: dict) -> None:
    """Why the model is GA2M and not just a GAM."""
    pairs = [n for n in ('age_years & engine_cc', 'mileage_km & engine_cc')
             if n in m.term_names_]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.4, 3.0))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, pairs):
        xe, ye, z = pair_grid(m, name)
        lim = float(np.nanmax(np.abs(z))) or 1e-6
        im = ax.pcolormesh(ye, xe, z, cmap=fs.DIVERGING,
                           norm=TwoSlopeNorm(0, -lim, lim), shading='flat')
        a, b = name.split(' & ')
        ax.set_xlabel(b)
        ax.set_ylabel(a)
        ax.set_title(name, fontsize=9.5)
        if 'engine_cc' in b:
            ax.set_xlim(500, 4000)
        if a == 'age_years':
            ax.set_ylim(0, 30)
        if a == 'mileage_km':
            ax.set_ylim(0, 300_000)
            ax.yaxis.set_major_formatter(FuncFormatter(fs.rupees))
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=7.5, labelcolor=fs.INK2)
        cb.set_label('extra effect', fontsize=8, color=fs.INK2)

    fig.tight_layout(w_pad=2.0)
    fs.title(fig, 'Pairwise terms: what the additive part misses',
             y=1.06, ax=axes[0])
    fs.save(fig, 'fig12_interactions', FIGS)
    cap['f12'] = f'{len(pairs)} pairwise surfaces shown'


def fig_local(m, df: pd.DataFrame, cap: dict) -> None:
    """One prediction, fully decomposed - what the front-end will show."""
    va = df[df['split_grouped'] == 'val'].copy()
    # A mid-market used car, and specifically one the model predicts about as well
    # as it predicts anything. Picking an arbitrary row first gave a Mercedes C220
    # that was 23% under-predicted, which demonstrates the decomposition fine but
    # invites the reader to judge the model off a single unlucky case. The honest
    # illustration is a *typical* prediction, so take the row whose error is
    # closest to the median error of its own segment.
    va['pred'] = m.predict(va[features.FEATURES])
    va['err'] = (10 ** va['pred'] - va['price']).abs() / va['price']
    pick = va[(va['condition'] == 'Used') & va['price'].between(5e6, 12e6)]
    row = pick.loc[[(pick['err'] - pick['err'].median()).abs().idxmin()]]

    local = m.explain_local(row[features.FEATURES])
    d = local.data(0)
    contrib = pd.DataFrame({'term': d['names'], 'score': d['scores']})
    contrib = contrib[contrib['score'].abs() > 1e-6]
    contrib['abs'] = contrib['score'].abs()
    top = contrib.nlargest(10, 'abs').sort_values('score')

    base = float(np.ravel(m.intercept_)[0])
    pred = float(m.predict(row[features.FEATURES])[0])
    actual = float(row['price'].iloc[0])

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    colors = [fs.RED if s < 0 else fs.BLUE for s in top['score']]
    ax.barh(np.arange(len(top)), top['score'], color=colors, height=0.66,
            linewidth=0)
    ax.set_yticks(np.arange(len(top)), top['term'], fontsize=8.5)
    ax.axvline(0, color=fs.BASE, linewidth=0.8)
    ax.set_xlabel('Effect on price')
    span = float(top['score'].abs().max()) * 1.45
    ax.set_xlim(-span, span)
    fs.multiplier_ticks(ax, 'x')
    fs.hgrid(ax)

    fs.corner(ax, f'{row["title"].iloc[0][:46]}\n'
                  f'base Rs {10 ** base / 1e6:.2f}M  ->  '
                  f'predicted Rs {10 ** pred / 1e6:.2f}M\n'
                  f'actual asking price Rs {actual / 1e6:.2f}M', loc='br')
    fs.title(fig, 'One prediction, decomposed into its terms', ax=ax)
    fs.save(fig, 'fig13_local_explanation', FIGS)
    cap['f13'] = (f'{row["title"].iloc[0]}: predicted Rs {10 ** pred:,.0f} '
                  f'vs actual Rs {actual:,.0f}')


def main() -> None:
    fs.style()
    m = joblib.load(MODEL)
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'vehicles.csv').merge(
        pd.read_csv(ROOT / 'data' / 'processed' / 'splits.csv'), on='id')
    print(f'loaded EBM with {len(m.term_names_)} terms; '
          f'base price Rs {10 ** float(np.ravel(m.intercept_)[0]):,.0f}\n')

    cap: dict = {}
    fig_importance(m, cap)
    fig_numeric_shapes(m, cap)
    fig_categoricals(m, cap)
    fig_interactions(m, cap)
    fig_local(m, df, cap)

    print('\nnumbers to quote in the captions')
    print('-' * 72)
    for k in sorted(cap):
        print(f'  {k}: {cap[k]}')


if __name__ == '__main__':
    main()
