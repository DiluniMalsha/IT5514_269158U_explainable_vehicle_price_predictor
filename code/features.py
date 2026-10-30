"""The canonical feature set.

`vehicles.csv` deliberately carries more columns than the model should see: raw
and grouped versions of the same categorical, identifiers for tracing a row back
to its listing, and listing-level fields kept for the report. Importing the lists
below is what stops a redundant or leaking column reaching the model by accident.

Exclusions and the reason for each
----------------------------------
`model_year`      Spearman correlation with `age_years` is exactly -1.000. The
                  same feature twice. EBM is additive, so including both splits
                  one effect across two terms and halves the apparent importance
                  of each - the feature ranking in the report would be wrong.
`model`           Fully contained in `make_model`, and ambiguous on its own
                  because 12 model names are reused across brands.
`make_model`      Superseded by `make_model_grp`; the ungrouped version has 555
                  levels, 398 of them with under 10 rows.
`town`,
`town_grp`        Nested inside `district_grp`: 226 towns, 67 after grouping,
                  sitting inside 23 districts. Fitting both levels splits one
                  location effect across two additive terms - the `model_year`
                  objection again. District is also the level at which the
                  sampling bias can be described (60.3% Colombo), whereas a town
                  term would fit micro-geography from thin cells.
`edition`,
`edition_grp`     Free text, not a category: 3,304 distinct values over 8,954
                  rows, and grouping rare levels leaves 57.4% in 'Other' plus
                  33.2% 'Unknown'. Replaced by the `trim_*` flags extracted from
                  its tokens, which is where the signal actually is.
`views`           Spearman with price is -0.021. It also describes how the advert
                  performed rather than what the vehicle is worth, and is only
                  known after the listing has been live.
`days_online`     Same: an advert attribute, and partly an *outcome* of the price
                  being wrong. A car still listed after 50 days is plausibly
                  overpriced, so training on it leaks the answer backwards.
`id`, `slug`,
`title`           Identifiers. `title` is especially dangerous: it contains the
                  brand, model, trim and year as free text, so a model with
                  access to it can reconstruct everything else.
`fetched_at`      Constant within a single snapshot.
`district`        Superseded by `district_grp`. Three districts - Kilinochchi
                  (6 rows), Mullativu (4) and Mannar (1) - fall under 10 rows,
                  and a term fitted from one row is noise wearing a place name.
                  All 25 districts are present. The ungrouped column stays in
                  the CSV precisely so the report can quantify that
                  under-representation rather than hide it.
`is_auth_dealer`  False on all 9,532 listings that carry it, absent on the other
                  387. ikman does not populate it for cars, so it is a constant.

`make_model_grp` carries an explicit 'Unknown' level (1.5% of rows): listings where
the seller chose ikman's 'Other Model' dropdown option instead of naming a model.
That is a statement that the model is unknown, not a model, so it is collapsed to one
level rather than left as a junk level per brand - 'Toyota Other Model' had held 26
unrelated cars from Rs 5.9m to Rs 499m. `brand_grp` still records the brand.

`brand_grp` is kept alongside `make_model_grp` even though brand is nested inside
it. They are not redundant after grouping: once a rare make_model collapses into
'Other', `brand_grp` is the only thing still saying it was a Mercedes rather than
a Micro.
"""

TARGET = 'price'

# Log-transform the target. Raw price has skewness 7.54 and kurtosis 95.58; after
# log10 those become 0.04 and 1.41. Without it, the squared-error objective is
# dominated by a handful of supercars and the model is effectively fitted to the
# top 1% of listings, which hold 12.4% of total market value.
LOG_TARGET = True

CATEGORICAL = [
    'brand_grp',
    'make_model_grp',
    'trim_grade',
    'condition',
    'transmission',
    'body',
    'fuel_type',
    'district_grp',
    'membership_level',
]

NUMERIC = [
    'age_years',
    'mileage_km',
    'engine_cc',
]

# Binary flags. `is_zero_mileage` separates "unregistered, genuinely 0 km" from a
# low reading, so the mileage shape function is not asked to treat 0 as a point on
# the same curve. `mileage_contradicts_condition` marks the 28 rows where the two
# fields disagree, letting the model discount them instead of us guessing which
# field is wrong.
FLAGS = [
    'trim_stated',
    'trim_performance',
    'trim_premium',
    'trim_equipment',
    'is_zero_mileage',
    'mileage_contradicts_condition',
    'is_verified',
    'is_member',
]

FEATURES = CATEGORICAL + NUMERIC + FLAGS

# Columns present in the CSV and deliberately not modelled. Listed explicitly so
# that a reviewer can see the choice was made rather than overlooked.
EXCLUDED = [
    'id', 'slug', 'title', 'fetched_at',
    'model_year', 'model', 'make_model', 'edition', 'edition_grp',
    'town', 'town_grp', 'district',
    'brand',
    'views', 'days_online',
]


def check(df) -> None:
    """Fail loudly if the CSV and this list have drifted apart."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f'vehicles.csv is missing expected columns: {missing}')

    constant = [c for c in FEATURES if df[c].nunique(dropna=False) < 2]
    if constant:
        raise ValueError(
            f'these features take only one value and cannot inform the model: '
            f'{constant}. Remove them from FEATURES rather than letting them '
            f'appear in the importance ranking with zero contribution.'
        )

    unaccounted = set(df.columns) - set(FEATURES) - set(EXCLUDED) - {TARGET}
    if unaccounted:
        raise KeyError(
            f'columns in vehicles.csv that are neither features nor explicitly '
            f'excluded: {sorted(unaccounted)}. Add each to FEATURES or EXCLUDED '
            f'in features.py so the decision is recorded.'
        )
