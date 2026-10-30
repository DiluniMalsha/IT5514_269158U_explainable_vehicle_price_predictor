"""The EDA figures

Run: python code/explore/make_figures.py
Output: figures/*.png (200 dpi, for the PDF) and figures/*.pdf (vector)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'processed' / 'vehicles.csv'
FIGS = ROOT / 'figures'

# Chart style and palette is in figstyle.py, imported by every figure script
# in this project.
from figstyle import (BLUE, ORANGE, RED, MID, SURFACE, INK, INK2, MUTED,
                     GRID, BASE, DIVERGING, style, vgrid, hgrid, corner,
                     rupees)
import figstyle as fs


def title(fig, text: str, y: float = 1.02, ax=None) -> None:
    fs.title(fig, text, y, ax)


def save(fig, name: str) -> None:
    fs.save(fig, name, FIGS)


# --- figures -------------------------------------------------------------------

def fig_target(df: pd.DataFrame, cap: dict) -> None:
    """Why the target is log-transformed. The most load-bearing figure here."""
    p = df['price']
    lp = np.log10(p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    a.hist(p / 1e6, bins=80, color=BLUE, linewidth=0)
    a.set_title('Asking price, as collected')
    a.set_xlabel('Price (Rs, millions)')
    a.set_ylabel('Listings')
    a.set_xlim(0, p.max() / 1e6)
    vgrid(a)
    corner(a, f'skewness {p.skew():.2f}\nkurtosis {p.kurtosis():.2f}')

    b.hist(lp, bins=60, color=BLUE, linewidth=0)
    b.set_title('After $\\log_{10}$ transform')
    b.set_xlabel('Price (Rs, log scale)')
    b.set_ylabel('Listings')
    b.set_xticks([5, 6, 7, 8, 9], ['100k', '1M', '10M', '100M', '1B'])
    vgrid(b)
    corner(b, f'skewness {lp.skew():.2f}\nkurtosis {lp.kurtosis():.2f}', loc='tl')

    title(fig, 'The target is strongly right-skewed; a log transform fixes it',
          y=1.06, ax=a)
    save(fig, 'fig1_target_distribution')
    cap['f1'] = (f'raw skew {p.skew():.2f} / kurtosis {p.kurtosis():.2f}; '
                 f'log10 skew {lp.skew():.2f} / kurtosis {lp.kurtosis():.2f}')


def fig_year_gap(df: pd.DataFrame, cap: dict) -> None:
    """The import suspension: a structural hole, not a sampling artefact."""
    vc = df['model_year'].dropna().astype(int).value_counts().sort_index()
    vc = vc[vc.index >= 2000]
    gap = [2020, 2021, 2022]
    colors = [ORANGE if y in gap else BLUE for y in vc.index]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(vc.index, vc.values, color=colors, width=0.78, linewidth=0)
    ax.set_xlabel('Model year')
    ax.set_ylabel('Listings')
    ax.set_ylim(0, vc.max() * 1.16)
    vgrid(ax)

    gap_n = int(vc.reindex(gap).fillna(0).sum())
    # Placed in clear space above the mid-2010s bars; the orange marks already say
    # which years are meant, so no leader line is needed.
    ax.text(2019.4, vc.max() * 0.40,
            f'2020–2022: {gap_n} listings in total\n(vehicle imports suspended)',
            ha='right', va='bottom', fontsize=8.5, color=INK2)
    peak = int(vc.idxmax())
    ax.text(peak, vc.max(), f'{vc.max():,}  ', ha='right', va='bottom',
            fontsize=8.5, color=INK2)

    title(fig, 'Model years 2020–2022 are nearly absent: imports were suspended',
          ax=ax)
    save(fig, 'fig2_model_year_gap')
    cap['f2'] = (f'2020-22 total {gap_n} listings vs {vc.get(2018, 0)} for 2018 '
                 f'and {vc.max():,} for {peak}')


def fig_numeric(df: pd.DataFrame, cap: dict) -> None:
    """The two dominant numeric drivers, and the shape of each relationship."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    for ax, col, xlabel, xmax in (
            (a, 'age_years', 'Vehicle age (years)', 35),
            (b, 'mileage_km', 'Odometer (km, thousands)', 300)):
        d = df[[col, 'price']].dropna()
        x = d[col] / (1000 if col == 'mileage_km' else 1)
        ax.scatter(x, d['price'], s=4, color=BLUE, alpha=0.10, linewidths=0)

        # Median per bin: the trend the eye should follow, not a regression line.
        bins = np.linspace(0, xmax, 16)
        mid = (bins[:-1] + bins[1:]) / 2
        med = [d['price'][(x >= lo) & (x < hi)].median()
               for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(mid, med, color=ORANGE, linewidth=2, marker='o', markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label='median price per bin')

        ax.set_yscale('log')
        ax.set_xlim(0, xmax)
        ax.set_ylim(1e5, 1e9)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(FuncFormatter(rupees))
        vgrid(ax)
        r = d.corr(method='spearman').iloc[0, 1]
        corner(ax, f'Spearman {r:+.3f}')

    a.set_ylabel('Price (Rs, log scale)')
    a.set_title('Price against age')
    b.set_title('Price against odometer')
    a.legend(loc='lower left')
    title(fig, 'Both relationships are negative and clearly non-linear',
          y=1.06, ax=a)
    save(fig, 'fig3_price_vs_numeric')
    cap['f3'] = ', '.join(
        f'{c} {df[["price", c]].dropna().corr(method="spearman").iloc[0, 1]:+.3f}'
        for c in ('age_years', 'mileage_km'))


def fig_corr(df: pd.DataFrame, cap: dict) -> None:
    """Feature-to-feature correlation, which bounds what an additive model may claim.

    Only the lower triangle is drawn. A correlation matrix is symmetric, so a full
    grid states every number twice and spends its most saturated colour on a
    diagonal of 1.000 that carries no information.
    """
    cols = ['price', 'age_years', 'mileage_km', 'engine_cc']
    labels = ['Price', 'Age\n(years)', 'Odometer\n(km)', 'Engine\n(cc)']
    m = df[cols].corr(method='spearman')

    sub = m.iloc[1:, :-1].to_numpy(dtype=float)
    sub = np.where(np.triu(np.ones_like(sub, dtype=bool), k=1), np.nan, sub)

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    im = ax.imshow(np.ma.masked_invalid(sub), cmap=DIVERGING,
                   norm=TwoSlopeNorm(0, -1, 1))
    ax.set_xticks(range(len(cols) - 1), labels[:-1], fontsize=8.5)
    ax.set_yticks(range(len(cols) - 1), labels[1:], fontsize=8.5)
    # A 2px surface gap between cells, rather than a border around each.
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.grid(which='major', visible=False)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=9,
                    color=SURFACE if abs(v) > 0.75 else INK)

    cb = fig.colorbar(im, ax=ax, shrink=0.72, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK2)
    cb.set_label('Spearman $\\rho$', fontsize=8.5, color=INK2)

    title(fig, 'Age and odometer are highly\ncorrelated with each other',
          y=1.10, ax=ax)
    save(fig, 'fig4_correlation')
    cap['f4'] = f'age vs mileage {m.loc["age_years", "mileage_km"]:.3f}'


def fig_condition(df: pd.DataFrame, cap: dict) -> None:
    """The dataset is not purely used cars. Two measures, so two panels."""
    order = [c for c in ('Used', 'Brand New', 'Reconditioned', 'Import')
             if c in set(df['condition'])]
    # Wrapped so the four tick labels cannot collide at this figure width.
    ticks = {'Used': 'Used', 'Brand New': 'Brand\nnew',
             'Reconditioned': 'Recondi-\ntioned', 'Import': 'Import'}
    n = df['condition'].value_counts().reindex(order)
    med = df.groupby('condition')['price'].median().reindex(order)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    pos = np.arange(len(order))
    lbl = [ticks[c] for c in order]

    a.bar(pos, n.values, color=BLUE, width=0.62, linewidth=0)
    a.set_xticks(pos, lbl, fontsize=8.5)
    a.set_ylabel('Listings')
    a.set_title('How many of each')
    a.set_ylim(0, n.max() * 1.14)
    vgrid(a)
    for x, v in zip(pos, n.values):
        a.text(x, v, f'{v / len(df) * 100:.0f}%', ha='center', va='bottom',
               fontsize=8.5, color=INK2)

    b.bar(pos, med.values / 1e6, color=BLUE, width=0.62, linewidth=0)
    b.set_xticks(pos, lbl, fontsize=8.5)
    b.set_ylabel('Median price (Rs, millions)')
    b.set_title('What each is worth')
    b.set_ylim(0, med.max() / 1e6 * 1.14)
    vgrid(b)
    for x, v in zip(pos, med.values / 1e6):
        b.text(x, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, color=INK2)

    title(fig, 'Four market segments, not one: 20% of listings are brand new',
          y=1.06, ax=a)
    save(fig, 'fig5_condition_mix')
    cap['f5'] = '; '.join(f'{c} {n[c]:,} ({n[c] / len(df) * 100:.1f}%) '
                          f'median Rs {med[c] / 1e6:.2f}m' for c in order)


def fig_district(df: pd.DataFrame, cap: dict) -> None:
    """Geographic concentration: the sampling bias to disclose."""
    vc = df['district'].value_counts().head(12)[::-1]
    colors = [ORANGE if d == 'Colombo' else BLUE for d in vc.index]

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.barh(np.arange(len(vc)), vc.values, color=colors, height=0.68, linewidth=0)
    ax.set_yticks(np.arange(len(vc)), vc.index, fontsize=8.5)
    ax.set_xlabel('Listings')
    # Headroom so the direct label sits inside the axes instead of overflowing.
    ax.set_xlim(0, vc.max() * 1.34)
    hgrid(ax)
    share = vc['Colombo'] / len(df) * 100
    ax.text(vc['Colombo'], len(vc) - 1, f'  {share:.1f}% of all listings',
            va='center', ha='left', fontsize=8.5, color=INK2)
    title(fig, 'Colombo dominates: this is really a Colombo-market model',
          ax=ax)
    save(fig, 'fig6_district_concentration')
    top3 = df['district'].value_counts().head(3)
    cap['f6'] = (f'Colombo {share:.1f}%, top 3 {top3.sum() / len(df) * 100:.1f}%, '
                 f'{df["district"].nunique()} districts present')


def fig_heaping(df: pd.DataFrame, cap: dict) -> None:
    """Asking prices are quoted in round numbers: a hard noise floor under RMSE."""
    steps = [(1_000_000, 'Rs 1M'), (500_000, 'Rs 500k'), (250_000, 'Rs 250k'),
             (100_000, 'Rs 100k'), (50_000, 'Rs 50k'), (25_000, 'Rs 25k'),
             (10_000, 'Rs 10k')]
    share = [(df['price'] % s == 0).mean() * 100 for s, _ in steps]

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    pos = np.arange(len(steps))
    ax.bar(pos, share, color=BLUE, width=0.62, linewidth=0)
    ax.set_xticks(pos, [lbl for _, lbl in steps], fontsize=8.5)
    ax.set_ylabel('Share of listings (%)')
    ax.set_xlabel('Price is an exact multiple of')
    ax.set_ylim(0, 108)
    vgrid(ax)
    for x, v in zip(pos, share):
        ax.text(x, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, color=INK2)
    # Top-left: the only corner the ascending bars leave clear.
    corner(ax, f'only {df["price"].nunique():,} distinct prices\n'
               f'across {len(df):,} listings', loc='tl')
    title(fig, 'Prices are quoted in round numbers, so the target is granular',
          ax=ax)
    save(fig, 'fig7_price_heaping')
    cap['f7'] = (f'{(df["price"] % 50_000 == 0).mean() * 100:.1f}% are multiples of '
                 f'Rs 50k; {df["price"].nunique():,} distinct prices')


def fig_brands(df: pd.DataFrame, cap: dict) -> None:
    """What the dataset is actually made of."""
    vc = df['brand'].value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.barh(np.arange(len(vc)), vc.values, color=BLUE, height=0.68, linewidth=0)
    ax.set_yticks(np.arange(len(vc)), vc.index, fontsize=8.5)
    ax.set_xlabel('Listings')
    ax.set_xlim(0, vc.max() * 1.06)
    hgrid(ax)
    cov = vc.sum() / len(df) * 100
    corner(ax, f'these 12 brands cover {cov:.1f}% of listings\n'
               f'({df["brand"].nunique()} brands in total)', loc='br')
    title(fig, 'The twelve most listed brands', ax=ax)
    save(fig, 'fig8_top_brands')
    cap['f8'] = f'top 12 brands cover {cov:.1f}%; {df["brand"].nunique()} brands total'


def main() -> None:
    style()
    df = pd.read_csv(CSV)
    features.check(df)
    print(f'loaded {len(df):,} rows x {df.shape[1]} columns\n')

    cap: dict = {}
    fig_target(df, cap)
    fig_year_gap(df, cap)
    fig_numeric(df, cap)
    fig_corr(df, cap)
    fig_condition(df, cap)
    fig_district(df, cap)
    fig_heaping(df, cap)
    fig_brands(df, cap)

    print('\nnumbers to quote in the captions')
    print('-' * 72)
    for k in sorted(cap):
        print(f'  {k}: {cap[k]}')


if __name__ == '__main__':
    main()
