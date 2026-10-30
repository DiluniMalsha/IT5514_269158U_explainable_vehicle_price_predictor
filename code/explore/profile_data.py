""" Profile the cleaned dataset before any modelling.

Run: python code/explore/profile_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'processed' / 'vehicles.csv'

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 60)


def rule(title: str) -> None:
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def main() -> None:
    df = pd.read_csv(CSV)
    print(f'loaded {len(df):,} rows x {df.shape[1]} columns from {CSV.name}')

    rule('1. THE EXTREMES - are they real cars or junk?')
    cols = ['title', 'price', 'condition', 'age_years', 'mileage_km', 'district']
    print('\n-- 12 highest prices --')
    print(df.nlargest(12, 'price')[cols].to_string(index=False))
    print('\n-- 12 lowest prices --')
    print(df.nsmallest(12, 'price')[cols].to_string(index=False))

    rule('2. TARGET DISTRIBUTION - does it need a log transform?')
    p = df['price']
    print(f'  count      {len(p):>14,}')
    print(f'  mean       {p.mean():>14,.0f}')
    print(f'  median     {p.median():>14,.0f}')
    print(f'  std        {p.std():>14,.0f}')
    print(f'  skewness   {p.skew():>14.2f}   (0 = symmetric; >1 = strong right tail)')
    print(f'  kurtosis   {p.kurtosis():>14.2f}')
    print(f'  mean/median{p.mean() / p.median():>14.2f}')
    lp = np.log10(p)
    print(f'\n  after log10:')
    print(f'  skewness   {lp.skew():>14.2f}')
    print(f'  kurtosis   {lp.kurtosis():>14.2f}')

    print('\n  percentiles:')
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0):
        print(f'    {q * 100:>5.1f}%  Rs {p.quantile(q):>14,.0f}')

    share = p.nlargest(int(len(p) * 0.01)).sum() / p.sum() * 100
    print(f'\n  top 1% of listings hold {share:.1f}% of total market value')

    rule('3. CATEGORICAL CARDINALITY - what can EBM actually learn?')
    for c in ['brand', 'model', 'make_model', 'condition', 'transmission',
              'body', 'fuel_type', 'district', 'town']:
        if c not in df.columns:
            continue
        n = df[c].nunique()
        # How many levels have too few rows to estimate a reliable term?
        vc = df[c].value_counts()
        rare = int((vc < 10).sum())
        covered = vc[vc >= 10].sum() / len(df) * 100
        print(f'  {c:<12} {n:>5} levels | {rare:>4} with <10 rows | '
              f'{covered:>5.1f}% of rows in levels with >=10')

    rule('4. SMALL CATEGORICALS - full breakdown')
    for c in ['condition', 'transmission', 'fuel_type', 'body']:
        if c not in df.columns:
            continue
        print(f'\n-- {c} --')
        vc = df[c].value_counts()
        med = df.groupby(c)['price'].median()
        tbl = pd.DataFrame({'n': vc, 'pct': (vc / len(df) * 100).round(1),
                            'median_price': med.round(0)})
        print(tbl.sort_values('n', ascending=False).to_string())

    rule('5. GEOGRAPHIC CONCENTRATION - sampling bias check')
    vc = df['district'].value_counts()
    print(f'  {df["district"].nunique()} districts present')
    print(f'  top district: {vc.index[0]} with {vc.iloc[0]:,} rows '
          f'({vc.iloc[0] / len(df) * 100:.1f}%)')
    print(f'  top 3 districts hold {vc.head(3).sum() / len(df) * 100:.1f}% of rows')
    print('\n' + vc.head(12).to_string())

    rule('6. TOP BRANDS')
    g = df.groupby('brand').agg(n=('price', 'size'), median_price=('price', 'median'),
                                median_age=('age_years', 'median'))
    print(g.sort_values('n', ascending=False).head(15).round(0).to_string())

    rule('7. NUMERIC FEATURES')
    num = ['price', 'age_years', 'mileage_km', 'engine_cc', 'views', 'days_online']
    num = [c for c in num if c in df.columns]
    print(df[num].describe().T.to_string(float_format=lambda v: f'{v:,.1f}'))

    rule('8. CORRELATION WITH PRICE (Spearman - rank based, robust to the skew)')
    for c in num:
        if c == 'price':
            continue
        r = df[['price', c]].dropna().corr(method='spearman').iloc[0, 1]
        print(f'  {c:<14} {r:>6.3f}')

    rule('9. SUSPICIOUS PATTERNS')
    zero_mil = (df['mileage_km'] == 0).sum()
    print(f'  mileage exactly 0 km:        {zero_mil:>6,} ({zero_mil / len(df) * 100:.1f}%)')
    for cond in df['condition'].unique():
        sub = df[df['condition'] == cond]
        z = (sub['mileage_km'] == 0).sum()
        print(f'    of which condition={cond:<14} {z:>5,} / {len(sub):,}')

    # Round-number prices are a tell for "asking price", not transaction price.
    round_m = (df['price'] % 1_000_000 == 0).sum()
    round_100k = (df['price'] % 100_000 == 0).sum()
    print(f'\n  price a whole million:       {round_m:>6,} ({round_m / len(df) * 100:.1f}%)')
    print(f'  price a multiple of 100k:    {round_100k:>6,} '
          f'({round_100k / len(df) * 100:.1f}%)')

    dup_slug = df.duplicated(subset=['title', 'price', 'mileage_km']).sum()
    print(f'\n  same title+price+mileage:    {dup_slug:>6,} '
          '(possible relisted duplicates)')

    rule('10. DATA QUALITY SUMMARY FOR THE REPORT')
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(f'  rows: {len(df):,}   columns: {df.shape[1]}')
    print(f'  columns with missing values: {len(miss)}')
    for c, n in miss.sort_values(ascending=False).items():
        print(f'    {c:<14} {n:>6,} ({n / len(df) * 100:.1f}%)')


if __name__ == '__main__':
    main()
