"""Streamlit front-end for the explainable vehicle price model.

Run: streamlit run app/app.py
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
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# features and figstyle are the shared modules at the top of code/; the shape
# function reader lives with the script that draws the report's explainability
# figures, one folder deeper, so both directories go on the path.
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'explain'))

import features                                    # noqa: E402
import figstyle as fs                              # noqa: E402
from explain_figures import continuous_shape       # noqa: E402

import trim_rules                                  # noqa: E402

# Must run before any other Streamlit call
st.set_page_config(page_title='Vehicle price explainer', layout='wide')

MODEL_PATH = ROOT / 'models' / 'ebm_grouped.joblib'
DATA_PATH = ROOT / 'data' / 'processed' / 'vehicles.csv'

# The year the dataset was collected. `clean_data.py` computed age_years as
# snapshot_year - model_year, so the interface has to use the same constant.
SNAPSHOT_YEAR = 2026

# Median absolute percentage error by price band, measured once on the held-out
# test set (report Table 12). Shown against whichever band an estimate lands in,
# because a single headline error figure would be misleading by a factor of four.
BANDS = [
    (0, 2e6, 'below Rs 2M', 26.3),
    (2e6, 5e6, 'Rs 2–5M', 13.7),
    (5e6, 1e7, 'Rs 5–10M', 6.5),
    (1e7, 2e7, 'Rs 10–20M', 6.1),
    (2e7, 5e7, 'Rs 20–50M', 10.7),
    (5e7, float('inf'), 'above Rs 50M', 9.7),
]

LABELS = {
    'brand_grp': 'Brand',
    'make_model_grp': 'Make and model',
    'trim_grade': 'Trim grade',
    'condition': 'Condition',
    'transmission': 'Transmission',
    'body': 'Body',
    'fuel_type': 'Fuel',
    'district_grp': 'District',
    'membership_level': 'Seller membership',
    'age_years': 'Age',
    'mileage_km': 'Odometer',
    'engine_cc': 'Engine capacity',
    'trim_stated': 'Trim stated',
    'trim_performance': 'Performance trim',
    'trim_premium': 'Premium trim',
    'trim_equipment': 'Equipment package',
    'is_zero_mileage': 'Zero odometer',
    'mileage_contradicts_condition': 'Odometer conflicts with condition',
    'is_verified': 'Verified seller',
    'is_member': 'Member seller',
}


# --------------------------------------------------------------------- loading

@st.cache_resource
def load():
    """Model, dataset and the vocabulary the model actually understands.

    The dropdown options come from the model's own bins rather than from the
    CSV. A level the model never saw would be routed to its 'unseen' bin and
    scored zero, which is a silent wrong answer rather than a visible one.
    """
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    levels = {}
    for i, name in enumerate(model.feature_names_in_):
        mapping = model.bins_[i][0]
        if isinstance(mapping, dict):
            levels[name] = sorted(mapping)

    known = set(levels['make_model_grp'])
    brand_models = {}
    for brand, group in data.groupby('brand_grp')['make_model_grp']:
        counts = group[group.isin(known)].value_counts()
        if counts.empty:
            continue
        # Most listed first, so the default selection is a real model rather
        # than whatever sorts earliest: alphabetically that is 'Other', which
        # would open the app on a collapsed level and an immediate warning.
        named = [m for m in counts.index if m not in ('Other', 'Unknown')]
        collapsed = [m for m in counts.index if m in ('Other', 'Unknown')]
        brand_models[brand] = named + collapsed

    # Which model years each model actually appears in. The year input used to
    # default to a fixed 2018 regardless of the model chosen, and because the
    # model defaults to the brand's most listed one, the app opened on a 2018
    # Toyota Raize - a vehicle that does not exist, since the Raize launched in
    # late 2019. Two defaults picked independently produced an impossible car.
    years = {}
    for mm, group in data.groupby('make_model_grp')['model_year']:
        counts = group.dropna().astype(int).value_counts()
        if counts.empty:
            continue
        # A single listing at some year is usually a typing slip rather than
        # evidence the model was sold then, so the quoted range ignores those.
        solid = sorted(y for y, n in counts.items() if n >= 2)
        years[mm] = {
            # The most listed year, not the median: a median can land in a gap
            # between two clusters, which would open the app on a year the model
            # was never sold in and fire the warning below against its own
            # default. The mode is always a year that exists.
            'default': int(counts.idxmax()),
            'present': set(counts.index),
            'low': solid[0] if solid else int(counts.index.min()),
            'high': solid[-1] if solid else int(counts.index.max()),
        }
    return model, data, levels, brand_models, years


def band_for(price: float):
    for lo, hi, label, err in BANDS:
        if lo <= price < hi:
            return label, err
    return BANDS[-1][2], BANDS[-1][3]


def pretty_term(name: str, values: dict) -> str:
    """Term name plus the value it was evaluated at, e.g. 'Age — 8 years'."""
    if ' & ' in name:
        a, b = name.split(' & ')
        return f'{LABELS.get(a, a)} × {LABELS.get(b, b)}'

    label = LABELS.get(name, name)
    v = values.get(name)
    if name == 'age_years':
        return f'{label} — {int(v)} years'
    if name == 'mileage_km':
        return f'{label} — {int(v):,} km'
    if name == 'engine_cc':
        return f'{label} — not stated' if v is None or pd.isna(v) else f'{label} — {int(v):,}cc'
    if name in ('trim_stated', 'trim_performance', 'trim_premium', 'trim_equipment',
                'is_zero_mileage', 'mileage_contradicts_condition',
                'is_verified', 'is_member'):
        return f'{label} — {"yes" if v else "no"}'
    return f'{label} — {v}'


# ----------------------------------------------------------------------- input

model, data, levels, brand_models, model_years = load()
base_log = float(np.ravel(model.intercept_)[0])
base_price = 10 ** base_log

st.title('Explainable Vehicle Price Estimator')
st.caption(
    'Explainable Boosting Machine, 35 terms, fitted on 8,954 ikman.lk listings. '
    'The estimate is the base price multiplied by one factor per feature, and '
    'every factor below is read directly off the model.'
)

sb = st.sidebar
sb.header('Vehicle')

brand = sb.selectbox('Brand', levels['brand_grp'],
                     index=levels['brand_grp'].index('Toyota')
                     if 'Toyota' in levels['brand_grp'] else 0)

options = brand_models.get(brand) or levels['make_model_grp']
if 'Other' in levels['make_model_grp'] and 'Other' not in options:
    options = options + ['Other']
make_model = sb.selectbox('Model', options,
                          help='"Other" is the level the model fits for make/model '
                               'combinations too rare to earn their own term.')

# The default year follows the model, so changing the model cannot leave a year
# behind that the model was never sold in. Keying the widget on the model name
# is what makes Streamlit re-initialise it: without the key it would keep
# whatever value it already held.
seen = model_years.get(make_model)
model_year = sb.slider('Model year', 1950, SNAPSHOT_YEAR + 1,
                       seen['default'] if seen else 2018,
                       key=f'model_year::{make_model}')
age_years = max(0, SNAPSHOT_YEAR - model_year)

# Condition and odometer follow the model and year for the same reason the year
# follows the model: three defaults chosen independently describe a vehicle
# nobody owns. A 2026 model with a fixed 85,000 km on it is the obvious case.
like_this = data[(data['make_model_grp'] == make_model)
                 & data['model_year'].between(model_year - 2, model_year + 2)]
typical_condition = (like_this['condition'].mode().iloc[0]
                     if len(like_this) >= 3 else 'Used')
typical_mileage = (int(round(like_this['mileage_km'].median() / 1000) * 1000)
                   if len(like_this) >= 3 and like_this['mileage_km'].notna().any()
                   else 85_000)

condition = sb.selectbox('Condition', levels['condition'],
                         index=levels['condition'].index(typical_condition),
                         key=f'condition::{make_model}::{model_year}')
transmission = sb.selectbox('Transmission', levels['transmission'])
body = sb.selectbox('Body', levels['body'],
                    index=levels['body'].index('Hatchback')
                    if 'Hatchback' in levels['body'] else 0)
fuel_type = sb.selectbox('Fuel type', levels['fuel_type'],
                         index=levels['fuel_type'].index('Petrol')
                         if 'Petrol' in levels['fuel_type'] else 0)

mileage_km = sb.number_input('Odometer (km)', min_value=0, max_value=755_000,
                             value=typical_mileage, step=1_000,
                             key=f'mileage::{make_model}::{model_year}')

# An electric vehicle has no engine capacity to state. The cleaner blanked the
# field for those rows rather than imputing a number, and the model learned a
# score for the missing bin, so the interface has to be able to send a blank.
no_engine = sb.checkbox('No engine capacity (electric)', value=fuel_type == 'Electric')
engine_cc = None if no_engine else sb.number_input(
    'Engine capacity (cc)', min_value=600, max_value=7_100, value=1_500, step=100)

district = sb.selectbox('District', levels['district_grp'],
                        index=levels['district_grp'].index('Colombo')
                        if 'Colombo' in levels['district_grp'] else 0)

sb.header('Trim')
edition = sb.text_input(
    'Trim or edition, as the seller would write it', value='',
    placeholder='e.g. G grade safety package',
    help='Free text. The flags below are extracted from it with the same '
         'patterns used to build the training data.')
trim = trim_rules.extract(edition)

for key, description in trim_rules.LABELS.items():
    mark = '✓' if trim[key] else '·'
    sb.caption(f'{mark}  {description}')
sb.caption(f'{"✓" if trim["trim_stated"] else "·"}  Trim stated at all '
           f'(grade: {trim["trim_grade"]})')

sb.header('Seller')
membership = sb.selectbox('Membership level', levels['membership_level'],
                          index=levels['membership_level'].index('free'))
is_verified = sb.checkbox('Verified seller', value=False)
# is_member is not an independent choice: 8,938 of 8,954 listings agree with
# membership_level, so letting the two be set separately would invite an input
# combination the model has essentially never seen.
is_member = int(membership != 'free')

values = {
    'brand_grp': brand,
    'make_model_grp': make_model,
    'trim_grade': trim['trim_grade'],
    'condition': condition,
    'transmission': transmission,
    'body': body,
    'fuel_type': fuel_type,
    'district_grp': district,
    'membership_level': membership,
    'age_years': float(age_years),
    'mileage_km': float(mileage_km),
    'engine_cc': np.nan if engine_cc is None else float(engine_cc),
    'trim_stated': trim['trim_stated'],
    'trim_performance': trim['trim_performance'],
    'trim_premium': trim['trim_premium'],
    'trim_equipment': trim['trim_equipment'],
    'is_zero_mileage': int(mileage_km == 0),
    'mileage_contradicts_condition': int(mileage_km == 0 and condition == 'Used'),
    'is_verified': int(is_verified),
    'is_member': is_member,
}

row = pd.DataFrame([values])[features.FEATURES]


# ------------------------------------------------------------------- predicting

pred_log = float(model.predict(row)[0])
pred_price = 10 ** pred_log
band_label, band_err = band_for(pred_price)

local = model.explain_local(row).data(0)
contrib = pd.DataFrame({'term': local['names'], 'score': local['scores']})
contrib = contrib[contrib['score'].abs() > 1e-9].copy()
contrib['multiplier'] = 10 ** contrib['score']
contrib = contrib.reindex(contrib['score'].abs().sort_values(ascending=False).index)

peers = data[(data['make_model_grp'] == make_model)
             & data['model_year'].between(model_year - 2, model_year + 2)]
peer_median = float(peers['price'].median()) if len(peers) >= 3 else None

headline = st.columns(3, gap='large')
headline[0].metric('Estimated asking price', f'Rs {pred_price:,.0f}')
headline[1].metric(f'Typical error, {band_label}', f'±{band_err:.1f}%',
                   help='Median absolute percentage error for this price band, '
                        'measured once on the held-out test set.')
headline[2].metric('Same-model median',
                   f'Rs {peer_median:,.0f}' if peer_median else 'too few listings',
                   help='What a person computes by hand: the median asking price '
                        'of other listings of this model within ±2 model years.')

left, right = st.columns([1, 1.6], gap='large')

with left:
    st.write(
        f'Typical error for listings **{band_label}** is **±{band_err:.1f}%** '
        f'of asking price, which on this estimate is roughly '
        f'Rs {pred_price * band_err / 100:,.0f}. '
        f'That figure is the median absolute error measured once on the '
        f'held-out test set, not on the data the model was fitted to.'
    )

    # The baseline a person computes by hand. If the model cannot beat it, the
    # model is not earning its complexity, so it belongs on the same screen.
    if peer_median is not None:
        st.write(
            f'The same-model median is drawn from **{len(peers)} listings** of '
            f'{make_model} within ±2 model years. Matching on model alone '
            f'reaches R² 0.537 on the test set against this model’s 0.9386; '
            f'matching the year as well closes most of that gap on the typical '
            f'vehicle, but still misses by more than half on roughly one '
            f'listing in ten, against one in thirty here, because a lookup has '
            f'nothing to fall back on where the same model and year are thinly '
            f'listed. Treat it as a sanity check rather than a rival estimate.'
        )
    else:
        st.write(
            f'The dataset holds only **{len(peers)} listing(s)** of {make_model} '
            f'within ±2 model years, so there is no reliable same-model median '
            f'to check this estimate against.'
        )

    warnings = []
    # The combination, not just each field on its own. A model that was never
    # sold in the chosen year is the one input that looks completely ordinary
    # and is in fact impossible.
    if seen and model_year not in seen['present']:
        warnings.append(
            f'The dataset holds no {make_model} from {model_year}. Listings of '
            f'this model run from {seen["low"]} to {seen["high"]}, so this '
            f'estimate rests on the general age curve rather than on evidence '
            f'about this model at this age.'
        )
    if 2020 <= model_year <= 2022:
        warnings.append(
            'Sri Lanka suspended vehicle imports from 2020, so model years '
            '2020–2022 hold only 159 listings in total. The age effect here is '
            'interpolated between the years on either side rather than learned.'
        )
    if make_model in ('Other', 'Unknown'):
        warnings.append(
            f'"{make_model}" is a collapsed level, not a specific model: the '
            'estimate rests on the brand term and the vehicle\'s attributes.'
        )
    if district == 'Other':
        warnings.append(
            'This district was too thinly represented to fit its own term. '
            '60.3% of the dataset is Colombo, so location effects outside the '
            'Western Province are weakly supported.'
        )
    if pred_price < 2e6:
        warnings.append(
            'Below Rs 2M the model is at its weakest, with a median error of '
            '26.3%. Accident history, service records and mechanical condition '
            'are not in the data and dominate a small absolute price.'
        )
    if mileage_km > 300_000:
        warnings.append(
            'Few listings report over 300,000 km, so the odometer curve is '
            'poorly supported at this reading.'
        )
    # The two engine inputs can be set to disagree, and the resulting estimate
    # looks confident while resting on almost no data. Say so rather than
    # quietly returning it.
    if no_engine and fuel_type != 'Electric':
        warnings.append(
            f'Engine capacity is blank in only 159 listings, almost all of them '
            f'electric. A blank capacity on a {fuel_type.lower()} vehicle is a '
            f'combination the model has barely seen, and the missing-value bin '
            f'(×0.74) will dominate the estimate.'
        )
    if fuel_type == 'Electric' and not no_engine:
        warnings.append(
            'Electric listings had their engine capacity blanked during '
            'cleaning rather than imputed, so an electric vehicle with a stated '
            'capacity is outside the data the model was fitted to.'
        )
    for w in warnings:
        st.warning(w)

with right:
    st.subheader('How the model reached that figure')
    st.write(
        f'The model is fitted on log₁₀(price), so the terms **multiply**. '
        f'It starts from a base of **Rs {base_price:,.0f}** — the average '
        f'listing in the dataset — and applies one factor per feature.'
    )

    shown = contrib.head(10)
    rest = contrib.iloc[10:]
    rest_mult = float(np.prod(rest['multiplier'])) if len(rest) else 1.0

    fs.style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    plot = shown.iloc[::-1]
    colors = [fs.RED if s < 0 else fs.BLUE for s in plot['score']]
    ax.barh(np.arange(len(plot)), plot['score'], color=colors, height=0.66,
            linewidth=0)
    ax.set_yticks(np.arange(len(plot)),
                  [pretty_term(t, values) for t in plot['term']], fontsize=8.5)
    ax.axvline(0, color=fs.BASE, linewidth=0.8)
    ax.set_xlabel('Effect on price')
    span = float(shown['score'].abs().max()) * 1.45 or 0.1
    ax.set_xlim(-span, span)
    fs.multiplier_ticks(ax, 'x')
    fs.hgrid(ax)
    fs.title(fig, 'The ten largest terms for this vehicle', ax=ax)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

st.markdown('##### The same decomposition as arithmetic')
st.caption(
    'Each factor is applied in turn, largest first. Because the contributions '
    'are additive in log space, the running product is exact at every step: '
    'this is the arithmetic the model performed, not an attribution that only '
    'approximately sums to the output.'
)

walk = [{'Step': 'Base — the average listing in the dataset', 'Factor': '—',
         'Running estimate': f'Rs {base_price:,.0f}'}]
running = base_price
for _, r in shown.iterrows():
    running *= r['multiplier']
    walk.append({'Step': pretty_term(r['term'], values),
                 'Factor': f'×{r["multiplier"]:.3f}',
                 'Running estimate': f'Rs {running:,.0f}'})
if len(rest):
    running *= rest_mult
    walk.append({'Step': f'All {len(rest)} remaining terms combined',
                 'Factor': f'×{rest_mult:.3f}',
                 'Running estimate': f'Rs {running:,.0f}'})
walk.append({'Step': 'Estimated asking price', 'Factor': '',
             'Running estimate': f'Rs {pred_price:,.0f}'})
# Sized to the row count so the whole walk is visible at once. Left to its
# default the frame scrolls internally, which hides the last line - and the last
# line is the point: the running product lands exactly on the estimate.
st.dataframe(pd.DataFrame(walk), hide_index=True, width='stretch',
             height=(len(walk) + 1) * 35 + 3)

st.divider()
st.subheader('Where this vehicle sits on the fitted curves')
st.write(
    'These are not plots about the model, they are slices of it. The shaded '
    'band is one standard deviation across the 14 bagged fits, and it widens '
    'where the data thin out — that is the model showing which parts of its own '
    'curve to trust. A value of ×1.00 is the dataset average, not "no effect".'
)

panels = [('age_years', float(age_years), f'{age_years} years'),
          ('mileage_km', float(mileage_km), f'{mileage_km:,} km')]
if engine_cc is not None:
    panels.append(('engine_cc', float(engine_cc), f'{engine_cc:,}cc'))

cols = st.columns(len(panels), gap='medium')
for col, (name, value, shown_value) in zip(cols, panels):
    edges, body, sd, _miss, _miss_sd = continuous_shape(model, name)
    centres = (edges[:-1] + edges[1:]) / 2
    with col:
        fs.style()
        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        ax.fill_between(centres, body - sd, body + sd, color=fs.BLUE, alpha=0.16,
                        linewidth=0)
        ax.plot(centres, body, color=fs.BLUE, linewidth=1.6)
        ax.axhline(0, color=fs.BASE, linewidth=0.8)
        here = float(np.interp(value, centres, body))
        ax.plot([value], [here], marker='o', markersize=6, color=fs.ORANGE,
                zorder=5)
        ax.annotate(f'{shown_value}\n×{10 ** here:.2f}', (value, here),
                    textcoords='offset points', xytext=(8, 8), fontsize=8,
                    color=fs.ORANGE)
        ax.set_xlabel(LABELS[name])
        fs.multiplier_ticks(ax, 'y')
        fs.vgrid(ax)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

with st.expander('About the model, and what it cannot do'):
    st.markdown(
        """
**Measured once on a held-out test set of 1,368 listings** that no tuning
decision was allowed to see.

| Model | R² | Median error | 90th-percentile error |
|---|---|---|---|
| Predict the global median | −0.003 | 33.6% | 138.7% |
| Same-model median lookup | 0.537 | 11.9% | 83.9% |
| Ridge regression | 0.895 | 10.7% | 33.6% |
| Random forest, 400 trees | 0.9339 | 6.4% | 28.0% |
| **This model (EBM, 35 terms)** | **0.9386** | 7.6% | **26.4%** |

The EBM and the random forest are not reliably distinguishable: a bootstrap 95%
confidence interval on the R² difference is [−0.006, +0.014] and contains zero.
The forest is more accurate on the typical vehicle; this model makes fewer large
mistakes. It was chosen because every factor shown above is read directly off
the fitted model rather than estimated from it afterwards.

**Limits worth knowing before trusting a number.**

* The target is an **asking price**, not a sale price. The estimate answers
  "what would this be listed at", and asking prices carry a negotiation margin
  that is not in the data.
* Inverting a log-space prediction estimates the **median** rather than the
  mean, so estimates are slightly conservative and should not be summed to
  value a fleet.
* Age and odometer correlate at 0.883, so the odometer curve cannot be read as
  the effect of mileage with age held constant.
* **60.3% of the data is Colombo.** This is a Colombo-market model with thin
  coverage elsewhere.
* Accident history, service records, owner count and auction grade are not on
  any public listing and so are not in the model. That is the real ceiling on
  accuracy, and it is why errors are worst on inexpensive older vehicles.
        """
    )
