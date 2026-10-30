"""Pre-processing Step 2: fix the train/validation/test split once, and save it.

Every later script reads `data/processed/splits.csv` rather than splitting again.
Reason: if each model split the data itself, a difference in scores
could be a difference in luck, and the comparison in the report would be worthless.

Hyperparameters are chosen on validation; the test set is
scored once, at the end. A test set that has been looked at repeatedly has been
trained on, slowly, through the choices it informed.

Two split strategies are produced side by side:

`split_random`   Ordinary stratified split. The number a reader expects.
`split_grouped`  Listings describing the same model, year and specification are
                 kept together. Exact duplicate feature vectors were already
                 dropped in cleaning, but 13% of rows sit in groups of three or
                 more near-identical cars priced within 10% of each other - one
                 dealer's showroom stock. Under a random split those near-twins
                 land on both sides and the test score is flattered.

Reporting both, and the gap between them, is more honest than picking one. The
gap is itself a finding about the data.

Both are stratified on **price decile crossed with condition**. Price decile alone
is not enough: because `condition` is one of the grouping keys, the grouped split
is free to push whole showrooms of brand-new cars into one fold, and it did - a
first attempt left the test set 23.5% brand new against the validation set's
16.5%. Brand-new listings are the easy ones (a showroom list price is nearly
deterministic), so that imbalance alone would have made test look easier than
validation and the comparison meaningless. Crossing the two brings the brand-new
share to 20.4 / 20.2 / 19.6 across train / validation / test.

Stratifying on price also keeps the 123 listings above Rs 100m from landing
unevenly, which would otherwise make the test score partly a measure of which
supercars happened to fall where.

Run: python code/preprocessing/split_data.py
Output: data/processed/splits.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data' / 'processed' / 'vehicles.csv'
OUT = ROOT / 'data' / 'processed' / 'splits.csv'

SEED = 42
N_FOLDS = 7          # fold 0 -> test, fold 1 -> validation, folds 2-6 -> train
N_PRICE_BINS = 10

# What makes two listings "the same car" for the purposes of the grouped split.
# Deliberately excludes price: grouping on the target would leak it.
GROUP_KEYS = ['make_model', 'model_year', 'condition',
              'transmission', 'body', 'fuel_type']


def assign(folds: list[np.ndarray], n: int) -> np.ndarray:
    """Fold 0 becomes test, fold 1 validation, the rest train."""
    out = np.empty(n, dtype=object)
    out[folds[0]] = 'test'
    out[folds[1]] = 'val'
    for f in folds[2:]:
        out[f] = 'train'
    return out


def report(df: pd.DataFrame, col: str) -> None:
    print(f'\n--- {col} ---')
    g = df.groupby(col, sort=False)
    tbl = pd.DataFrame({
        'rows': g.size(),
        'share': (g.size() / len(df) * 100).round(1),
        'median_price': g['price'].median().round(0),
        'p05': g['price'].quantile(0.05).round(0),
        'p95': g['price'].quantile(0.95).round(0),
        'above_100m': g['price'].apply(lambda s: int((s >= 1e8).sum())),
        'brand_new': g['condition'].apply(lambda s: int((s == 'Brand New').sum())),
    }).reindex(['train', 'val', 'test'])
    print(tbl.to_string())


def main() -> None:
    df = pd.read_csv(CSV)
    n = len(df)
    print(f'loaded {n:,} rows')

    price_bin = pd.qcut(df['price'], N_PRICE_BINS, labels=False, duplicates='drop')
    groups = df[GROUP_KEYS].astype(str).agg('|'.join, axis=1).factorize()[0]
    print(f'{len(set(groups)):,} distinct (model, year, spec) groups; '
          f'largest holds {pd.Series(groups).value_counts().iloc[0]} rows')

    # Stratum = price decile x condition. Any stratum too small to appear in every
    # fold is merged back into its price decile, so the fold count stays valid and
    # the fallback is explicit rather than a warning nobody reads.
    strata = price_bin.astype(str) + '|' + df['condition'].astype(str)
    counts = strata.value_counts()
    thin = set(counts[counts < N_FOLDS].index)
    if thin:
        strata = strata.where(~strata.isin(thin), price_bin.astype(str) + '|merged')
        print(f'{len(thin)} thin strata merged into their price decile '
              f'({int(counts[list(thin)].sum())} rows)')
    strata = pd.factorize(strata)[0]
    print(f'{len(set(strata))} strata, smallest holds '
          f'{pd.Series(strata).value_counts().iloc[-1]} rows')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rand_folds = [te for _, te in skf.split(np.zeros(n), strata)]

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    grp_folds = [te for _, te in sgkf.split(np.zeros(n), strata, groups)]

    out = pd.DataFrame({
        'id': df['id'],
        'group_id': groups,
        'price_bin': price_bin,
        'split_random': assign(rand_folds, n),
        'split_grouped': assign(grp_folds, n),
    })

    # --- integrity checks: loud failure beats a quietly wrong split -----------
    for col in ('split_random', 'split_grouped'):
        assert out[col].notna().all(), f'{col} has unassigned rows'
        assert set(out[col]) == {'train', 'val', 'test'}, f'{col} has odd labels'
    assert out['id'].is_unique, 'duplicate ids - splits cannot be joined back'

    # The grouped split only means something if no group straddles it.
    straddle = (out.groupby('group_id')['split_grouped'].nunique() > 1).sum()
    assert straddle == 0, f'{straddle} groups straddle the grouped split'
    straddle_rand = (out.groupby('group_id')['split_random'].nunique() > 1).sum()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    chk = df.join(out.drop(columns='id'))
    report(chk, 'split_random')
    report(chk, 'split_grouped')

    print(f'\ngroups straddling the split: grouped 0 (enforced), '
          f'random {straddle_rand:,} of {len(set(groups)):,}')
    print('  that number is exactly the leakage the grouped split removes.')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
