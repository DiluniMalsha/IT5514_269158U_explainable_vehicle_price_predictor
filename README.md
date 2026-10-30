# Explainable Used Vehicle Price Prediction for the Sri Lankan Market

**IT5514 Applied Machine Learning — Final Assignment**
W. M. D. M. Peiris · Index No. 269158U · MSc in Artificial Intelligence, University of Moratuwa

Predicts the **asking price** of a vehicle advertised for sale in Sri Lanka from
its observable attributes, and explains every prediction. The dataset is 8,954
listings compiled from ikman.lk. The model is an **Explainable Boosting Machine**
(a generalized additive model with pairwise interactions), chosen because it is
interpretable by construction rather than by post-hoc approximation — the
prediction *is* the explanation: a base price multiplied by one factor per
feature, read directly off the fitted model.

## Result

Measured once on a held-out test set of 1,368 listings that no tuning decision
was allowed to see.

| Model | R² | RMSE (log₁₀) | Median error | 90th-pct error |
|---|---|---|---|---|
| Predict the global median | −0.003 | 0.379 | 33.6% | 138.7% |
| Same-model median lookup | 0.537 | 0.258 | 11.9% | 83.9% |
| Ridge regression | 0.895 | 0.123 | 10.7% | 33.6% |
| Random forest, 400 trees | 0.9339 | 0.0974 | **6.4%** | 28.0% |
| **EBM, 35 terms** | **0.9386** | **0.0938** | 7.6% | **26.4%** |

The EBM and the random forest are **not reliably distinguishable**: a bootstrap
95% confidence interval on the R² difference is [−0.006, +0.014] and contains
zero, and the forest was never tuned while nine EBM configurations were evaluated
on validation. The defensible claim is comparable accuracy with different error
profiles — the forest is better on the typical vehicle, the EBM makes fewer large
mistakes. The argument for the EBM is not that it wins; it is that reading every
term off the model cost nothing measurable in fit.

A single R² also hides a fourfold spread: median error runs from 6.1% in the
Rs 10–20M band to 26.3% below Rs 2M, where accident history and mechanical
condition — which no public listing carries — dominate a small absolute price.

## Layout

```
├── code/
│   ├── features.py              the canonical feature set, and a check() that
│   │                            fails if a column is neither feature nor
│   │                            declared exclusion
│   ├── figstyle.py              shared chart style and validated palette
│   ├── ikman.py                 fetch helpers, rate limiting, JSON extraction
│   ├── scrape/                  scrape_index.py, scrape_details.py
│   ├── preprocessing/           clean_data.py, split_data.py
│   ├── explore/                 profile_data.py, make_figures.py
│   ├── train/                   baselines.py, train_ebm.py
│   ├── test/                    evaluate_test.py
│   └── explain/                 explain_figures.py, shap_comparison.py
├── app/                         Streamlit front-end
├── data/raw/                    9,949 advertisements, 9,919 detail pages
├── data/processed/              vehicles.csv (the submitted dataset) and scores
├── models/                      the fitted EBMs
├── figures/                     every figure in the report, PNG and PDF
```

The three modules at the top of `code/` are shared by more than one stage; every
other script belongs to exactly one. Because each stage script sits one directory
below `code/`, two lines at the top of each put the shared modules on the import
path and locate the project root, so any script runs directly from anywhere with
no package installation and no environment variable to set.

## Running it

```bash
python -m pip install -r requirements.txt
```

The pipeline, in order. Each stage writes a file the next one reads, so the whole
result rebuilds from the raw collection. Stages 1–2 are only needed to collect
the data again; `data/raw/` already holds the collection the report describes.

```bash
python code/scrape/scrape_index.py                 # -> data/raw/index.jsonl
python code/scrape/scrape_details.py               # -> data/raw/details.jsonl
python code/preprocessing/clean_data.py            # -> data/processed/vehicles.csv
python code/preprocessing/split_data.py            # -> data/processed/splits.csv
python code/explore/profile_data.py                # printed profile
python code/explore/make_figures.py                # -> figures/fig1..8
python code/train/baselines.py                     # -> data/processed/baseline_scores.csv
python code/train/train_ebm.py                     # -> models/ebm_*.joblib
python code/explain/explain_figures.py             # -> figures/fig9..13
python code/explain/shap_comparison.py             # -> figures/fig14, shap_comparison.csv
python code/test/evaluate_test.py                  # the single test-set scoring
```

The canonical reproduction of the headline numbers is one command:

```bash
python code/train/train_ebm.py --interactions 15 --learning-rate 0.06 --test
```

### The front-end

Run it from this folder, not from `app/`, so Streamlit picks up
`.streamlit/config.toml` — the light theme that matches the report figures.

```bash
streamlit run app/app.py
```

It opens on <http://localhost:8501>. The user describes a vehicle and the app
returns an estimate **with the decomposition that produced it**. Three things are
on screen that a demo would be tempted to hide:

* **the typical error for the price band the estimate falls into**, measured on
  the test set, because one headline error figure would be wrong by a factor of
  four;
* **the same-model median** — the baseline a person computes by hand, and the bar
  the model has to clear;
* **warnings when the inputs fall where the data are thin**: the 2020–2022
  import-suspension gap, collapsed `Other` levels, the sub-Rs 2M band, odometer
  readings above 300,000 km, and engine/fuel combinations the model has barely
  seen.

The decomposition is shown as a chart and again as arithmetic, walking from the
base price to the estimate one factor at a time. The running product is exact at
every step: it is the arithmetic the model performed, not an attribution that
approximately sums to the output.

## Two details that will bite if they are changed

**`SNAPSHOT_YEAR = 2026` in `app/app.py` is deliberate.** `clean_data.py` computed
`age_years` as `snapshot_year - model_year`, with `snapshot_year` taken from the
clock at collection time. If the app read the current year instead of the
constant, then from January 2027 every age sent to the model would be one year
out and the estimates would be silently wrong.

**`app/trim_rules.py` duplicates the cleaner's regular expressions.** The four
`trim_*` flags and `trim_grade` are tokens extracted from free text, so the app
must derive them exactly as the training data did. The patterns are a verbatim
copy of the corresponding block in `code/preprocessing/clean_data.py`. They are
copied rather than imported because that script's output is what the report
certifies as reproducible, and refactoring it after the dataset was frozen risks
changing that output. The copy was verified against all 8,954 rows: **0
mismatches across all five derived fields.** If the cleaner's patterns change,
change them here too.

## Data and ethics

Collected from ikman.lk on 25 October 2026 between 10:56 and 15:28 UTC. The
submitted dataset is `data/processed/vehicles.csv`, reproduced exactly by running
`code/preprocessing/clean_data.py` over the two files in `data/raw/`.

* **No personal or sensitive data was collected.** Seller names, account
  identifiers, shop names, telephone numbers and profile images were all present
  in the source and were deliberately excluded. No field in any of the three
  files identifies a seller.
* **`robots.txt` was honoured**, including its prohibition on the telephone-number
  endpoint — the one field that would have made this a collection of personal
  data.
* **Bot protection was respected rather than circumvented.** Two candidate sites
  blocked automated access and were abandoned.
* **Server load was minimised:** rate limiting with jitter, backoff on errors, at
  most three concurrent requests, each page fetched once, resumable runs.
* **A methodological cost was accepted for privacy.** Discarding seller identity
  means dealer clustering cannot be tested and a seller-grouped split is
  impossible; grouping by vehicle specification is the substitute, and the
  trade-off is disclosed in the report rather than omitted.
