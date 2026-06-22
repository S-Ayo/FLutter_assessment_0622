# Flutterwave Risk Analytics Assessment

## What this is

This is my submission for the Flutterwave Risk Analytics take-home assessment. The task was to build a fraud detection model on a synthetic transaction dataset, write a production SQL query for feature extraction, and put together a business strategy for managing the precision-recall tradeoff.

Everything runs from a single Jupyter notebook (`fraud_detection.ipynb`). The business strategy is in a separate document as requested. This file explains my thinking and approach.

---

## Files

```
fraud_detection.ipynb          — main notebook covering Parts 1 and 2
business_strategy_summary.md   — Part 3 business strategy write-up
README.md                      — this file
create_notebook.py             — script I used to generate the notebook programmatically
```

Running the notebook also produces these charts:
`eda_class_distribution.png`, `eda_categorical.png`, `eda_time.png`, `confusion_matrix.png`, `roc_pr_curves.png`, `threshold_analysis.png`, `feature_importance.png`

---

## Setup

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook fraud_detection.ipynb
```

Run all cells top to bottom. The notebook reads `flutterwave_synthetic_txs (1).csv` from the same directory and saves charts alongside it.

---

## The Dataset

52,500 transactions across 9 columns. A few things I noticed immediately when I loaded it:

- The `timestamp` column has two different formats mixed in (`2026-05-24T07:56:29Z` and `2026-05-06 21:49:19`). I handled this early with `pd.to_datetime(format='mixed', utc=True)`.
- Fraud rate is 1.9% — heavily imbalanced, which shaped the entire modeling approach.
- No missing values anywhere, which is unusually clean for a real dataset.

---

## Part 1 — SQL

The query is in the notebook under Part 1. It's written for PostgreSQL / Redshift and computes two rolling features per transaction, using only data that existed at the time of each transaction (no lookahead).

**Feature 1 — `user_24h_amount_usd`:** Rolling 24-hour sum of a user's prior spending. This was straightforward using a window function with `RANGE BETWEEN '24 hours' PRECEDING AND '1 microsecond' PRECEDING`. The microsecond offset excludes the current row without needing `EXCLUDE CURRENT ROW` (which isn't supported in all Redshift versions).

**Feature 2 — `user_1h_distinct_countries`:** Distinct card countries used by the same user in the past hour. PostgreSQL doesn't support `COUNT(DISTINCT ...)` inside RANGE window frames, so I used a correlated subquery instead. This is fine at moderate scale but needs a composite index on `(user_id, timestamp)` to be practical — without that index it's a full scan per row.

For production at scale I'd pre-materialise these features into a micro-batch table that refreshes every few minutes rather than recomputing the window functions on the full history every time.

---

## Part 2 — Modeling

### Data cleaning

Besides the timestamp format issue, the main thing I had to deal with was `dispute_status`. When I cross-tabbed it against `is_fraud` I found that two of the values — `pending` and `chargeback_won` — are 100% correlated with fraud. These statuses only get assigned after a chargeback is processed, so they don't exist at the moment a transaction is being scored. Including them as model features would produce great-looking metrics that collapse to zero in production.

I handled this by keeping `dispute_status` out of the model entirely and using it as a post-model override instead (explained at the end of the notebook). If a transaction is in the `pending` or `chargeback_won` state when it hits the decision layer, it gets flagged regardless of what the model scored.

### Feature engineering

I built 20 features across five areas:

**Time** — hour of day, day of week, weekend flag, night flag (11pm–6am), business hours flag. Fraudsters sometimes have different activity patterns to legitimate users so these felt worth including.

**Amount signals** — raw amount plus a z-score relative to that user's own transaction history. The idea is that a $300 transaction from someone who usually spends $20 is more suspicious than the same $300 from someone who regularly spends that much.

**Rolling window features** — these mirror the SQL query from Part 1: 24h cumulative spend, 24h transaction count, and 1h distinct countries. I recomputed them in Python using `pandas.rolling('24h', closed='left')` for the sum/count, and a per-user numpy loop for the distinct country count (pandas doesn't support `nunique()` with time-based rolling windows).

**Device signals** — how many distinct users have used this device. Devices shared across many users can indicate fraud rings or credential stuffing.

**User behaviour** — total transaction count, spend standard deviation, number of distinct countries ever used, and a flag for whether this is the first time the user has transacted from this card country.

### Model choice

I went with XGBoost for a few reasons. It handles mixed feature types without needing normalisation, has a `scale_pos_weight` parameter that directly addresses the class imbalance, and the feature importance scores are easy to explain to a non-technical stakeholder.

For the imbalance I set `scale_pos_weight = 51.5` (ratio of legitimate to fraud in the training set). I didn't use SMOTE or undersampling — the scale weight approach is simpler and works well with tree-based models.

80/20 train/test split, stratified on the target. Early stopping on PR-AUC.

### Why PR-AUC and not ROC-AUC?

At a 98:2 class ratio, ROC-AUC is misleading. A model that predicts every transaction as legitimate would score around 0.5 ROC-AUC and look reasonable. PR-AUC has a baseline equal to the fraud rate (~0.019), so any real lift shows up directly. I used it as both the training eval metric and the primary reported metric.

---

## Key Findings

### The model can't predict fraud from these features — and that's a data problem, not a modeling problem

After building and running the model, the results were poor:

- PR-AUC: 0.0205 vs. a random baseline of 0.0190 (1.08× lift — essentially nothing)
- ROC-AUC: 0.4752, which is below 0.5

I went back and checked whether this was a modeling issue, and it isn't. The features are simply not correlated with the fraud label. Statistical testing confirms this:

| Feature | Correlation with `is_fraud` | p-value |
|---|---|---|
| `dispute_status` (encoded) | 0.344 | — (post-hoc label) |
| `day_of_week` | −0.005 | 0.235 |
| `amount_usd` | −0.004 | 0.337 |
| `card_country` | 0.004 | — |
| `merchant_category` | −0.003 | — |
| `hour` | −0.001 | 0.870 |

None of the p-values are significant. The feature importances in the model also came out nearly equal across all 17 features (~6–7% each), which is the signature of a model fitting noise rather than signal.

I also compared the 660 confirmed fraud cases (backed by a chargeback) against the 340 unconfirmed ones — same amounts, same times, same categories. They're indistinguishable.

My read is that the fraud labels in this dataset were assigned randomly at a flat 1.9% rate, independent of the transaction characteristics. There are no patterns to find in the features provided.

The only strong predictor is `dispute_status`, which has a correlation of 0.344 — about 85× stronger than the next best feature — but as discussed, it's a retrospective label and can't be used at scoring time.

### What this means

The approach here — rolling velocity features, device signals, behavioural baselines, XGBoost with class weights, threshold-based tiers — is the right one for real fraud detection. On actual transaction data where fraud genuinely correlates with velocity spikes, device anomalies, and geographic inconsistencies, these features would be meaningfully predictive. The current result reflects the synthetic data generation process, not a flaw in the methodology.

---

## Design decisions worth explaining

**Why not use SMOTE?** SMOTE generates synthetic minority samples by interpolating between existing fraud cases. On a dataset where fraud labels are random and uncorrelated with features, SMOTE would just add noise. Even on real data I'd try class weights first before resorting to oversampling.

**Why exclude `dispute_status` from features entirely?** The temptation is to include it and get good-looking metrics. But in production every new transaction starts with `dispute_status = none`. A model trained on `pending` and `chargeback_won` would work perfectly in testing and fail silently the moment it hit live traffic. The override approach captures the value of this signal where it legitimately exists, without poisoning the model.

**Why recompute the SQL features in Python?** The SQL query defines what the production data pipeline outputs. The Python implementation uses the same window semantics (`closed='left'` for no lookahead) so the model trains on features that exactly match what it will receive in deployment. Inconsistency between training features and production features is one of the most common causes of model degradation.

---

## What I'd do differently with more time

A few things I'd explore:

- Build a user-device bipartite graph and extract graph centrality features to detect fraud rings more directly
- Use TargetEncoder instead of LabelEncoder for the categorical features — it handles unseen values better and captures the actual fraud rate per category
- Apply probability calibration (Platt scaling) so the model's output scores are interpretable as genuine probabilities rather than arbitrary scores
- On real data, compute user profile features (mean spend, std spend) from a 30-day rolling window rather than the full dataset, to avoid lookahead bias in training
