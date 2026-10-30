"""Pre-processing Step 1: turn the raw scrape into a modelling table.

Run:    python code/preprocessing/clean_data.py
Input:  data/raw/details.jsonl
Output: data/processed/vehicles.csv
"""

from __future__ import annotations

import re
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared modules live in code/, one level up from this script; data/ and
# models/ live in the project root, two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ikman

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw' / 'details.jsonl'
OUT = ROOT / 'data' / 'processed' / 'vehicles.csv'

# ---------------------------------Sanity bounds---------------------------------
#
# MAX_PRICE was first set to Rs 150m, which looked generous and was badly wrong:
# it silently deleted 27 genuine listings, including a Ferrari 296 GTB at Rs 255m,
# a Rolls-Royce Ghost at Rs 300m and a Lamborghini Temerario at Rs 400m. Sri Lankan
# import duty puts real supercars far above any threshold that "feels" high, so the
# bound has to come from the market rather than from intuition.
#
# Rs 500m was the second attempt and was still wrong, for the same reason: on the
# full 9,919-ad scrape it deleted a Rolls-Royce Phantom Extended at Rs 550m and a
# Phantom V12 at Rs 600m, both genuine, while happily keeping a "Toyota tiida 2008"
# listed at Rs 499m. That is the lesson learned: a univariate price bound
# cannot separate a real supercar from an extra-zero typo, because the two overlap
# in absolute price. The bound is therefore set only to catch impossible values,
# and the actual typo detection is done conditionally - see `peer_ratio` below.
MAX_PRICE = 750_000_000
#
# Prices below Rs 150k are not cheap cars: the six found were values like "Rs 192"
# for a 2026 Honda Vezel and "Rs 345" for a Land Rover Discovery. Read as lakhs
# those are Rs 19.2m and Rs 34.5m, both plausible, so they are unit-entry errors.
# There is no safe way to infer intent, so they are dropped rather than rescaled.
MIN_PRICE = 150_000
#
# km; beyond this the odometer reading is not credible
MAX_MILEAGE = 1_000_000

MIN_YEAR = 1950
MAX_ENGINE_CC = 8_000


def parse_number(value, suffix_pattern: str = '') -> float | None:
    """Pull an integer out of strings like 'Rs 8,700,000', '35,000 km', '1,000 cc'.

    Returns None rather than raising, so one odd listing cannot kill the run.
    """
    if value is None:
        return None
    s = str(value)
    if suffix_pattern:
        s = re.sub(suffix_pattern, '', s, flags=re.I)
    digits = re.sub(r'[^\d.]', '', s)
    if not digits or digits == '.':
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def main() -> None:
    rows = ikman.read_jsonl(RAW)
    if not rows:
        raise SystemExit(f'nothing at {RAW} - run scrape_details.py first')

    df = pd.DataFrame(rows)
    report: list[tuple[str, int]] = [('rows read from raw scrape', len(df))]

    # ------------------------ repair mis-decoded text -------------------------
    # The first scrape read every page as ISO-8859-1,
    # so UTF-8 bytes arrived re-interpreted one byte at a time: 'Coupe/Sports'
    # stored as 'CoupA(c)/Sports', '4x4' as '4A-4', the degree sign as 'A°'.
    # ikman.get now pins the encoding, so fresh scrapes are clean; this repairs
    # what is already on disk rather than re-fetching 9,919 pages to fix 83 rows.
    #
    # The round trip is exact for this corruption, because the mangling is a pure
    # byte-level relabelling: encoding back to latin-1 recovers the original bytes
    # and decoding them as UTF-8 recovers the text. Strings that were never
    # mangled either fail to encode as latin-1 or decode back to themselves, so
    # the fallback leaves them untouched.
    def repair(value):
        if not isinstance(value, str):
            return value
        try:
            return value.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value

    text_cols = [c for c in df.columns if df[c].dtype == object]
    fixed = 0
    for col in text_cols:
        repaired = df[col].map(repair)
        # Compare only where a string was actually present. A plain `!=` counts
        # every missing cell as changed, because NaN != NaN - a first version of
        # this reported 4,405 repairs when the true figure is 83.
        both = df[col].notna() & repaired.notna()
        fixed += int((repaired[both] != df[col][both]).sum())
        df[col] = repaired
    report.append(('repaired: mis-decoded text values', fixed))

    # --- deduplicate -------------------------------------------------------
    # Promoted ads appear on several search pages. If duplicates survive into
    # the split, the same car sits in train and test, leaking test data and
    # inflating every score.
    before = len(df)
    df = df.drop_duplicates(subset='id', keep='first')
    report.append(('dropped: duplicate ad id', before - len(df)))

    # --- numeric fields ----------------------------------------------------
    df['price'] = df['price_raw'].map(lambda v: parse_number(v, r'^\s*rs\.?\s*'))
    df['mileage_km'] = df['prop_mileage'].map(lambda v: parse_number(v, r'\s*km\s*$'))
    df['engine_cc'] = df['prop_engine_capacity'].map(lambda v: parse_number(v, r'\s*cc\s*$'))
    df['model_year'] = df['prop_model_year'].map(parse_number)

    # --- target must exist and be plausible --------------------------------
    before = len(df)
    df = df[df['price'].notna()]
    report.append(('dropped: no parseable price', before - len(df)))

    before = len(df)
    df = df[(df['price'] >= MIN_PRICE) & (df['price'] <= MAX_PRICE)]
    report.append((f'dropped: price outside {MIN_PRICE:,}-{MAX_PRICE:,}', before - len(df)))

    # --- year --------------------------------------------------------------
    snapshot_year = datetime.now().year
    before = len(df)
    df = df[df['model_year'].between(MIN_YEAR, snapshot_year + 1)]
    report.append((f'dropped: year outside {MIN_YEAR}-{snapshot_year + 1}', before - len(df)))

    # Age is the feature the model should see, not the raw year. A 2015 car means
    # something different in 2026 than it did in 2018, and age extrapolates to
    # future data where an unseen year value would not.
    #
    # Next model year vehicles are genuinely on sale before the year turns, so a
    # raw subtraction gives a negative age (one 2027 Mercedes in this snapshot).
    # Clipped to zero: a single row should not define the left-hand edge of the
    # age shape function, and "newer than new" is not a distinction the market makes.
    df['age_years'] = (snapshot_year - df['model_year']).clip(lower=0)

    # --- mileage -----------------------------------------------------------
    before = len(df)
    df = df[df['mileage_km'].isna() | (df['mileage_km'] <= MAX_MILEAGE)]
    report.append((f'dropped: mileage above {MAX_MILEAGE:,} km', before - len(df)))

    # A genuine 0 km reading means an unregistered brand-new vehicle, which the
    # condition field already tells us. Keep it, but flag it, because 0 is not
    # "unknown" and the shape function for mileage should not treat it as such.
    df['is_zero_mileage'] = (df['mileage_km'] == 0).astype(int)

    before = len(df)
    df = df[df['engine_cc'].isna() | (df['engine_cc'] <= MAX_ENGINE_CC)]
    report.append((f'dropped: engine above {MAX_ENGINE_CC:,} cc', before - len(df)))

    # Electric vehicles have no engine displacement, but the field is mandatory
    # on ikman's posting form, so sellers type something arbitrary - a BYD ATTO 3
    # in the sample carried "100 cc". Treating that as a real displacement would
    # have EBM fit a shape function over a meaningless axis and then invite us to
    # interpret it. Blank it and let the fuel_type term carry the information.
    electric = df['prop_fuel_type'].astype('string').str.strip().str.lower().eq('electric')
    n_blanked = int((electric & df['engine_cc'].notna()).sum())
    df.loc[electric, 'engine_cc'] = pd.NA
    report.append(('blanked: engine_cc on electric vehicles', n_blanked))

    # Implausibly small displacements on non-electric listings are the same kind
    # of form-filling noise (a 50 cc "car" does not exist).
    before_small = int((df['engine_cc'] < 500).sum())
    df.loc[df['engine_cc'] < 500, 'engine_cc'] = pd.NA
    report.append(('blanked: engine_cc below 500 cc', before_small))

    # --- categoricals ------------------------------------------------------
    cat_map = {
        'prop_brand': 'brand',
        'prop_model': 'model',
        'prop_edition': 'edition',
        'prop_condition': 'condition',
        'prop_transmission': 'transmission',
        'prop_body': 'body',
        'prop_fuel_type': 'fuel_type',
        'district': 'district',
        'town': 'town',
    }
    for src, dest in cat_map.items():
        col = df[src] if src in df.columns else pd.Series(index=df.index, dtype=object)
        # Normalise case and whitespace so 'Toyota ' and 'toyota' are one level.
        df[dest] = (col.astype('string').str.strip()
                    .replace({'': pd.NA})
                    .fillna('Unknown'))

    # brand+model together; 'Civic' alone is ambiguous across makes. 443 distinct
    # make_model values against 408 model values means ~35 model names are reused
    # across brands, so the combined field is the one to trust.
    # 'Other Model' is the dropdown option a seller picks when their car is not in
    # ikman's model list - it is a statement that the model is unknown, not a model.
    # Left alone it produces one junk level per brand: 'Toyota Other Model' held 26
    # unrelated cars from Rs 5.9m to Rs 499m, which is both a meaningless category
    # for the model to fit and a useless peer group for the price check below.
    # Collapsing them to 'Unknown' loses nothing, because `brand_grp` still carries
    # the brand - the same argument that justifies keeping brand alongside model.
    df['model'] = df['model'].replace({'Other Model': 'Unknown'})
    df['make_model'] = df['brand'].str.cat(df['model'], sep=' ')
    df.loc[df['model'].eq('Unknown'), 'make_model'] = 'Unknown'

    # Trim level. Present on ~70% of listings and a genuine price driver - a Yaris
    # "X Push Start" and a base Yaris are different cars.
    #
    # Treating it as a categorical does not work: 1,712 distinct values across
    # 3,764 rows, and grouping rare levels leaves 63.9% in 'Other' plus 31.0%
    # 'Unknown', so 95% of the column carries no information. It is free text that
    # sellers compose themselves, not a field with levels.
    #
    # What does work is pulling out the markers that recur across brands. Each
    # flag below is a thing a valuer would actually look for, and each is checked
    # against the token counts in the raw data rather than guessed. 'hybrid' is
    # deliberately not among them - fuel_type already carries it.
    df['edition'] = df['edition'].str.replace(r'\s+', ' ', regex=True)
    ed = df['edition'].astype('string').str.lower().fillna('')

    df['trim_stated'] = (~df['edition'].isin(['Unknown'])).astype(int)

    # Japanese-market grade letters (G/X/Z), the most common structured signal in
    # the field. Ordered roughly base -> mid -> high within a model line.
    grade = ed.str.extract(r'\b([gxz])\s*grade\b', expand=False)
    grade = grade.fillna(ed.str.extract(r'^([gxz])\b', expand=False))
    df['trim_grade'] = grade.str.upper().fillna('none')

    # Performance lines carry a large premium independent of engine size.
    df['trim_performance'] = ed.str.contains(
        r'\b(amg|gtx?|rs|sport|turbo|nismo|type\s*r|m\s*sport|st[- ]line)\b',
        regex=True).astype(int)

    # Luxury / top-of-range wording.
    df['trim_premium'] = ed.str.contains(
        r'\b(premium|luxury|highest|fully\s*loaded|autobiography|limited|exclusive)\b',
        regex=True).astype(int)

    # Equipment and safety packages.
    df['trim_equipment'] = ed.str.contains(
        r'\b(safety|bsm|led|4\s*way|360|camera|push\s*start|sunroof)\b',
        regex=True).astype(int)

    # --- dealer signals ----------------------------------------------------
    # Who is selling moves the price: verified sellers list at a visibly higher
    # median than unverified ones. These are listing attributes rather than
    # vehicle attributes, but unlike view counts they are known to a buyer at the
    # moment of valuation, so they are legitimate inputs rather than leakage.
    # isAuthDealer was tried here and dropped: it is False for every one of the
    # 3,851 listings that carry it and absent on the rest, so ikman does not
    # populate it for cars. A constant column cannot inform a model, and leaving
    # it in the feature set would put a zero-importance term in the report's
    # ranking as though it had been measured.
    for src, dest in (('isVerified', 'is_verified'),
                      ('isMember', 'is_member')):
        if src in df.columns:
            df[dest] = df[src].fillna(False).astype(int)
    if 'membershipLevel' in df.columns:
        df['membership_level'] = (df['membershipLevel'].astype('string')
                                  .str.strip().replace({'': pd.NA}).fillna('none'))

    # --- contradictory and placeholder readings -----------------------------
    # 10 listings describe a Used car with 0 km on the odometer. One of the two
    # fields is wrong and there is no way to tell which, so flag rather than guess.
    df['mileage_contradicts_condition'] = (
        (df['mileage_km'] == 0) & (df['condition'] == 'Used')
    ).astype(int)

    # Exactly 1,000,000 km is a keyboard placeholder, not an odometer reading.
    placeholder = df['mileage_km'] == MAX_MILEAGE
    report.append(('blanked: mileage exactly 1,000,000 km', int(placeholder.sum())))
    df.loc[placeholder, 'mileage_km'] = pd.NA

    # --- near-duplicate listings -------------------------------------------
    # Distinct ad ids can still describe the same physical vehicle: dealers relist,
    # and the earlier profile found 122 rows sharing title, price and mileage.
    # Identical feature vectors landing on both sides of the split let the model
    # memorise rather than generalise, so the test score stops measuring
    # generalisation. Two genuinely identical brand-new cars in one showroom would
    # also be collapsed here - that is the accepted cost of removing the leak.
    dup_keys = ['make_model', 'model_year', 'mileage_km', 'price',
                'condition', 'transmission', 'body', 'fuel_type']
    dup_keys = [c for c in dup_keys if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dup_keys, keep='first')
    report.append(('dropped: duplicate feature vector', before - len(df)))

    # --- conditional price-error detection ---------------------------------
    # A flat MAX_PRICE cannot find extra-zero typos, because a typo on a cheap car
    # lands in the same range as a genuine supercar: a Daihatsu Mira at Rs 76.5m
    # and a Ferrari 296 GTB at Rs 255m are both "high", but only one is wrong.
    # What separates them is the peer group. Compare each listing against other
    # listings of the SAME make_model within +/-2 model years. Conditioning on year
    # matters: 'Toyota Corolla' spans 1990s saloons at Rs 3m and 2026 Corolla Cross
    # at Rs 27m, so an unconditional median flags the new ones as errors - an
    # earlier version of this check did exactly that and produced 46 flags, most of
    # them real cars.
    #
    # The ratio distribution separates cleanly: 17 rows at >=4x the peer median, 16
    # of them at >=8x, and nothing in between that is ambiguous. Every one of the 16
    # is an order-of-magnitude slip (Mira Rs 76.5m against 241 same-era peers at
    # Rs 6.9m; Toyota Raize Rs 147m against 458 peers at Rs 13.7m). Genuine
    # supercars are untouched: they either have no peer group, or their peers are
    # priced like them.
    #
    # They are dropped rather than divided by 10. Rescaling assumes the error was
    # exactly one zero, and that cannot be verified from the listing.
    # Two windows, tried in order. A tight +/-2 years is the comparison we want, but
    # some model/year cells are nearly empty - a 2025 Wagon R Stingray listed at
    # Rs 88.5m had only three same-era peers and so escaped the check entirely while
    # an identical 2025 car sat in the data at Rs 9.9m. Falling back to +/-5 years
    # for those rows restores the comparison; five years of depreciation is well
    # inside the 5x threshold, so the wider window cannot manufacture a flag.
    PEER_WINDOWS = (2, 5)
    PEER_YEARS = PEER_WINDOWS[-1]   # widest window, used by the brand-level pass
    PEER_MIN = 3          # need a real peer group before trusting the comparison
    PEER_HIGH = 5.0       # above this, the price is not credible for the model/era
    PEER_LOW = 0.2        # below this it is usually a down-payment or lease figure
    PEER_LOW_MIN = 10     # the low side is noisier, so demand a bigger peer group

    ratios = pd.Series(np.nan, index=df.index, dtype=float)
    peer_n = pd.Series(0, index=df.index, dtype=int)
    for model, block in df.groupby('make_model', sort=False):
        # 'Unknown' is not a model - it is every car whose model the seller did not
        # state, pooled across all brands. Comparing a Rolls-Royce Phantom against
        # that pool deletes it as an outlier, which is the same mistake the flat
        # MAX_PRICE made. These rows are handled by the brand-level pass instead.
        if model == 'Unknown' or len(block) < PEER_MIN + 1:
            continue
        years = block['model_year'].to_numpy(dtype=float)
        prices = block['price'].to_numpy(dtype=float)
        for pos, idx in enumerate(block.index):
            if np.isnan(years[pos]):
                continue
            for window in PEER_WINDOWS:
                near = np.abs(years - years[pos]) <= window
                near[pos] = False
                if near.sum() < PEER_MIN:
                    continue
                med = np.median(prices[near])
                if med > 0:
                    ratios[idx] = prices[pos] / med
                    peer_n[idx] = int(near.sum())
                break

    # Rows with no usable model peer group get a second, coarser pass against the
    # same BRAND within the same era. This is what catches the worst single error in
    # the dataset: a 2008 listing titled "Toyota tiida" at Rs 499m, whose model the
    # seller left as 'Other Model', so it had no model peers at all. Against Toyotas
    # of its era it is ~80x the median. The threshold is looser than the model-level
    # one because a brand mixes segments - a Mercedes G800 AMG really is several
    # times the median Mercedes - and a 2x gap is not evidence of an error.
    # Restricted to rows whose model the seller never stated. That restriction is
    # what makes the pass safe. A first version applied it to every row without a
    # model peer group and deleted genuine halo models: a Nissan GT-R at Rs 195m is
    # really ~20x the median Nissan, and a Cadillac VISTIQ ~15x the median Cadillac.
    # Brand-relative outlier detection cannot tell a halo model from a typo, because
    # both sit far above their brand median - the same failure as the flat MAX_PRICE.
    # A listing with no stated model carries no claim to be a halo model, so for
    # those rows the brand median is the only evidence available and a 15x gap is
    # safe to treat as an error.
    BRAND_HIGH = 15.0
    brand_ratios = pd.Series(np.nan, index=df.index, dtype=float)
    unresolved = ratios.isna() & df['make_model'].eq('Unknown')
    for brand, block in df[unresolved].groupby('brand', sort=False):
        pool = df[df['brand'] == brand]
        if len(pool) < PEER_MIN + 1:
            continue
        pool_years = pool['model_year'].to_numpy(dtype=float)
        pool_prices = pool['price'].to_numpy(dtype=float)
        for idx, yr, price in zip(block.index, block['model_year'], block['price']):
            if pd.isna(yr):
                continue
            near = np.abs(pool_years - float(yr)) <= PEER_YEARS
            near &= pool.index.to_numpy() != idx
            if near.sum() < PEER_MIN:
                continue
            med = np.median(pool_prices[near])
            if med > 0:
                brand_ratios[idx] = price / med

    too_high = (ratios >= PEER_HIGH) | (brand_ratios >= BRAND_HIGH)
    too_low = (ratios <= PEER_LOW) & (peer_n >= PEER_LOW_MIN)
    before = len(df)
    df = df[~(too_high | too_low)]
    report.append((f'dropped: price >={PEER_HIGH:g}x same-model/era peer median',
                   int((ratios >= PEER_HIGH).sum())))
    report.append((f'dropped: price >={BRAND_HIGH:g}x same-brand/era median (model not stated)',
                   int((brand_ratios >= BRAND_HIGH).sum())))
    report.append((f'dropped: price <={PEER_LOW:g}x same-model/era peer median',
                   int(too_low.sum())))
    print(f'  [peer check] {int(ratios.notna().sum()):,} of {before:,} rows had a '
          f'usable peer group ({ratios.notna().mean() * 100:.1f}%)')

    # --- rare level grouping -----------------------------------------------
    # Run after deduplication so the counts describe the rows actually modelled.
    # EBM will fit a term for a category with three examples and that term is
    # noise, which then shows up in the feature-importance ranking as if it were
    # a finding. Collapsing the thin tail into 'Other' keeps the signal from the
    # well-populated levels and stops the model inventing the rest. A grouped
    # column is written alongside the original so the raw value stays available
    # for the report's tables.
    for col, min_count in (('make_model', 10), ('edition', 20),
                           ('town', 20), ('brand', 10), ('district', 10)):
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        rare = set(vc[vc < min_count].index) - {'Unknown'}
        grouped = f'{col}_grp'
        df[grouped] = df[col].where(~df[col].isin(rare), 'Other')
        n_rows = int(df[col].isin(rare).sum())
        report.append((f'grouped: {col} levels with <{min_count} rows '
                       f'({len(rare)} -> Other)', n_rows))

    # Listing-level, not vehicle-level. Kept for the discussion section but left
    # out of the default feature set: views and days-online describe how the
    # advert performed, not what the car is worth. Training on them answers a
    # different question than the one we are asking.
    df['views'] = pd.to_numeric(df.get('views'), errors='coerce')
    df['days_online'] = pd.to_numeric(df.get('postedDate'), errors='coerce')

    keep = [
        'id', 'slug', 'title',
        'price',
        'brand', 'brand_grp', 'model', 'make_model', 'make_model_grp',
        'edition', 'edition_grp',
        'trim_stated', 'trim_grade', 'trim_performance', 'trim_premium',
        'trim_equipment',
        'condition', 'transmission', 'body', 'fuel_type',
        'district', 'district_grp', 'town', 'town_grp',
        'model_year', 'age_years', 'mileage_km', 'engine_cc', 'is_zero_mileage',
        'mileage_contradicts_condition',
        'is_verified', 'is_member', 'membership_level',
        'views', 'days_online', 'fetched_at',
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # --- report ------------------------------------------------------------
    width = max(len(label) for label, _ in report) + 2
    print('\nCLEANING REPORT')
    print('=' * (width + 10))
    for label, n in report:
        print(f'{label:<{width}} {n:>7,}')
    print('-' * (width + 10))
    print(f'{"rows retained":<{width}} {len(out):>7,}')
    retained = len(out) / report[0][1] * 100 if report[0][1] else 0
    print(f'{"retention":<{width}} {retained:>6.1f}%')

    print('\nMISSING VALUES (retained rows)')
    miss = out.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        for c, n in miss.items():
            print(f'  {c:<18} {n:>6,}  ({n / len(out) * 100:.1f}%)')
    else:
        print('  none')

    print('\nTARGET (price, Rs)')
    print(out['price'].describe().apply(lambda v: f'{v:,.0f}').to_string())

    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
