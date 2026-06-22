"""
Generates fraud_detection.ipynb — the main assessment notebook.
Run: python3 create_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.0"
    }
}


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src):
    return nbf.v4.new_code_cell(src)


# ─────────────────────────────────────────────────────────────────────────────
# CELLS
# ─────────────────────────────────────────────────────────────────────────────

C00_title = md("""\
# Flutterwave Risk Analytics Assessment
## Fraud Detection: End-to-End Solution

| | |
|---|---|
| **Dataset** | `flutterwave_synthetic_txs (1).csv` — 52,500 synthetic transaction records |
| **Target** | `is_fraud` (1 = fraudulent chargeback, 0 = legitimate) |
| **Stack** | Python · XGBoost · scikit-learn · pandas · PostgreSQL/Redshift SQL |

---

### Structure
- **Part 1 — Data Engineering & SQL:** Production-ready rolling-window feature query (PostgreSQL / Redshift)
- **Part 2 — Fraud Modeling:** EDA → feature engineering → XGBoost → evaluation
- **Part 3 — Business Strategy:** See `business_strategy_summary.md`

---

> ### ⚠️ Key Data-Leakage Finding
> The `dispute_status` column contains values `pending` and `chargeback_won` that are
> **100% correlated** with `is_fraud = 1`.  These statuses are only assigned *after* fraud
> is confirmed — they do not exist at real-time scoring.
> **Strategy applied (Option C):** `dispute_status` is *excluded* from all model features
> but used as a **post-model secondary validation override** in the final decision layer.
""")

# ── SETUP ────────────────────────────────────────────────────────────────────

C01_setup = code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", 25)
pd.set_option("display.float_format", "{:.4f}".format)

DATA_PATH = "flutterwave_synthetic_txs (1).csv"
RANDOM_STATE = 42

print("Environment ready.")
""")

# ── PART 1 ───────────────────────────────────────────────────────────────────

C02_part1_header = md("""\
---
## Part 1 — Data Engineering & SQL

The query below is production-ready for **PostgreSQL ≥ 11** and **Amazon Redshift**.
It computes two rolling-window risk features *per transaction, at the exact moment the
transaction occurs* (i.e., only data strictly prior to `t` is used — no lookahead).
""")

C03_sql = code('''\
SQL_FEATURE_QUERY = """
-- ======================================================================
-- Production Feature Extraction Query
-- Engine : PostgreSQL ≥ 11  |  Amazon Redshift
-- Purpose: Compute rolling-window risk signals per transaction,
--          using only data that existed at the moment of each transaction
-- ======================================================================

WITH enriched AS (
    SELECT
        transaction_id,
        user_id,
        timestamp,
        amount_usd,
        card_country,
        merchant_category,
        device_id,
        dispute_status,
        is_fraud,

        -- ── FEATURE 1 ──────────────────────────────────────────────────
        -- Total amount_usd this user transacted in the PRECEDING 24 hours
        -- (current transaction excluded via microsecond offset)
        COALESCE(
            SUM(amount_usd) OVER (
                PARTITION BY user_id
                ORDER BY     timestamp
                RANGE BETWEEN \'24 hours\'::INTERVAL PRECEDING
                          AND \'1 microsecond\'::INTERVAL PRECEDING
            ),
        0) AS user_24h_amount_usd

    FROM transactions
),

one_hour_counts AS (
    -- ── FEATURE 2 ──────────────────────────────────────────────────────
    -- Distinct card_country values used by this user in the PRECEDING 1 hour.
    --
    -- COUNT(DISTINCT expr) is not supported inside RANGE window frames in
    -- PostgreSQL.  A correlated lateral subquery achieves the same result.
    -- Performance note: with a composite index on (user_id, timestamp) this
    -- subquery is an index-range-scan per row, NOT a full-table scan.
    SELECT
        e.transaction_id,
        COALESCE((
            SELECT COUNT(DISTINCT t2.card_country)
            FROM   transactions t2
            WHERE  t2.user_id   =  e.user_id
              AND  t2.timestamp <  e.timestamp
              AND  t2.timestamp >= e.timestamp - INTERVAL \'1 hour\'
        ), 0) AS user_1h_distinct_countries
    FROM enriched e
)

SELECT
    e.transaction_id,
    e.user_id,
    e.timestamp,
    e.amount_usd,
    e.card_country,
    e.merchant_category,
    e.device_id,
    e.dispute_status,
    e.is_fraud,
    e.user_24h_amount_usd,
    ohc.user_1h_distinct_countries
FROM       enriched          e
JOIN       one_hour_counts   ohc  USING (transaction_id)
ORDER BY   e.user_id, e.timestamp;
"""

print(SQL_FEATURE_QUERY)
''')

C04_sql_perf = md("""\
### SQL Performance Considerations

| Concern | Recommendation |
|---|---|
| **Composite index** | `CREATE INDEX idx_txn_user_time ON transactions(user_id, timestamp);` — both the window function and the correlated subquery exploit this B-tree path. |
| **Redshift distribution** | `DISTKEY(user_id)` co-locates every row for a user on the same node, making both operations fully node-local. Pair with `COMPOUND SORTKEY(user_id, timestamp)` for zone-map pruning. |
| **COUNT(DISTINCT) at scale** | The correlated subquery is O(n × k) where k = avg hourly transactions per user. For billion-row tables consider: (a) pre-aggregating into an hourly micro-batch summary materialised view, or (b) using `LISTAGG(DISTINCT card_country, ',')` + application-side dedup. |
| **Timestamp type** | Store as `TIMESTAMPTZ` (PostgreSQL) or UTC `TIMESTAMP` (Redshift) to avoid timezone edge cases in RANGE frames. |
| **NULL safety** | `COALESCE(…, 0)` ensures users with no prior transactions return 0 — critical for downstream model features. |
| **Incremental execution** | In production, scope the query to a micro-batch of new transactions joined against a pre-partitioned 24 h rolling buffer table rather than a full-table scan. |
| **Materialised view** | For batch scoring pipelines, pre-materialise these features hourly (with INCREMENTAL refresh) and invalidate on new data arrival to avoid repeated window recomputation. |
""")

# ── PART 2 ───────────────────────────────────────────────────────────────────

C05_part2_header = md("""\
---
## Part 2 — Fraud Modeling

### 2.1 Data Loading & Cleaning
""")

C06_load = code("""\
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape  : {df.shape}")
print(f"\\nColumn dtypes:")
print(df.dtypes)
print(f"\\nMissing values:")
print(df.isnull().sum())
print(f"\\nSample rows:")
df.head(3)
""")

C07_timestamps = code("""\
# Two timestamp formats co-exist in the raw data:
#   - ISO-8601 with Z suffix  → "2026-05-24T07:56:29Z"
#   - Plain space-separated   → "2026-05-06 21:49:19"
# pd.to_datetime with utc=True parses both and normalises to UTC-aware datetime.

fmt1 = df['timestamp'].str.contains('T').sum()
fmt2 = len(df) - fmt1
print(f"ISO-Z format   : {fmt1:,} rows")
print(f"Space format   : {fmt2:,} rows")

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
print(f"\\nAfter parsing  : {df['timestamp'].dtype}")
print(f"Date range     : {df['timestamp'].min().date()}  →  {df['timestamp'].max().date()}")
""")

C08_eda_fraud_dist = md("### 2.2 Exploratory Data Analysis")

C09_eda_class = code("""\
fraud_count = df['is_fraud'].sum()
total       = len(df)
fraud_rate  = fraud_count / total

print(f"Total transactions  : {total:,}")
print(f"Fraudulent          : {fraud_count:,} ({fraud_rate:.2%})")
print(f"Legitimate          : {total - fraud_count:,} ({1-fraud_rate:.2%})")
print(f"Imbalance ratio     : {(total - fraud_count)/fraud_count:.0f} : 1  (legit : fraud)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

counts = df['is_fraud'].value_counts()
axes[0].bar(['Legitimate', 'Fraudulent'], [counts[0], counts[1]],
            color=['steelblue', 'crimson'], alpha=0.85, edgecolor='black')
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Transaction count')
for i, v in enumerate([counts[0], counts[1]]):
    axes[0].text(i, v + 300, f'{v:,}\\n({v/total:.1%})', ha='center', va='bottom', fontsize=10)

df[df['is_fraud']==0]['amount_usd'].hist(bins=60, alpha=0.55, ax=axes[1],
    density=True, color='steelblue', label='Legitimate')
df[df['is_fraud']==1]['amount_usd'].hist(bins=60, alpha=0.55, ax=axes[1],
    density=True, color='crimson', label='Fraudulent')
axes[1].set_title('Amount Distribution by Class', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Amount (USD)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nAmount statistics by class:")
print(df.groupby('is_fraud')['amount_usd'].describe().round(2))
""")

C10_leakage = code('''\
print("=" * 62)
print("  DATA LEAKAGE ANALYSIS - dispute_status x is_fraud")
print("=" * 62)

leakage = (
    df.groupby('dispute_status')
    .agg(total=('is_fraud', 'count'), fraud_count=('is_fraud', 'sum'))
    .assign(fraud_rate=lambda x: x['fraud_count'] / x['total'])
)
print(leakage.round(4))

print("""
FINDING:
  - pending        : 307 rows,  100.0% fraud
  - chargeback_won : 353 rows,  100.0% fraud
  - inquiry        : 17,270 rows, 0.0% fraud  (issuer scrutiny, not confirmed)
  - none           : 34,570 rows, 1.0% fraud  (no action taken)

  These retrospective labels DO NOT exist at transaction time.
  Strategy (Option C):
    - Exclude from model features entirely
    - Use as post-model override in the final decision layer
""")
''')

C11_eda_cat = code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fraud rate by merchant category
cat_stats = (
    df.groupby('merchant_category')
    .agg(total=('is_fraud', 'count'), fraud=('is_fraud', 'sum'))
    .assign(fraud_rate=lambda x: x['fraud'] / x['total'])
    .sort_values('fraud_rate', ascending=True)
)
axes[0].barh(cat_stats.index, cat_stats['fraud_rate'],
             color='coral', edgecolor='black', alpha=0.85)
axes[0].set_title('Fraud Rate by Merchant Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Fraud Rate')
for i, (idx, row) in enumerate(cat_stats.iterrows()):
    axes[0].text(row['fraud_rate'] + 0.0005, i,
                 f'{row["fraud_rate"]:.2%}  (n={row["total"]:,})', va='center', fontsize=9)

# Fraud rate by card country
cc_stats = (
    df.groupby('card_country')
    .agg(total=('is_fraud', 'count'), fraud=('is_fraud', 'sum'))
    .assign(fraud_rate=lambda x: x['fraud'] / x['total'])
    .sort_values('fraud_rate', ascending=True)
)
axes[1].barh(cc_stats.index, cc_stats['fraud_rate'],
             color='steelblue', edgecolor='black', alpha=0.85)
axes[1].set_title('Fraud Rate by Card Country', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Fraud Rate')
for i, (idx, row) in enumerate(cc_stats.iterrows()):
    axes[1].text(row['fraud_rate'] + 0.0005, i,
                 f'{row["fraud_rate"]:.2%}  (n={row["total"]:,})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_categorical.png', dpi=150, bbox_inches='tight')
plt.show()

print("Merchant category stats:")
print(cat_stats.sort_values('fraud_rate', ascending=False))
print("\\nCard country stats:")
print(cc_stats.sort_values('fraud_rate', ascending=False))
""")

C12_eda_time = code("""\
df['hour']       = df['timestamp'].dt.hour
df['day_of_week']= df['timestamp'].dt.dayofweek  # 0 = Monday

hourly_rate = df.groupby('hour')['is_fraud'].mean()
dow_rate    = df.groupby('day_of_week')['is_fraud'].mean()
overall     = df['is_fraud'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(hourly_rate.index, hourly_rate.values, color='purple', alpha=0.75, edgecolor='black')
axes[0].axhline(y=overall, color='red', linestyle='--', lw=1.5,
                label=f'Overall avg ({overall:.3f})')
axes[0].set_title('Fraud Rate by Hour of Day (UTC)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Fraud Rate')
axes[0].legend()

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].bar(range(7), dow_rate.values, color='teal', alpha=0.75, edgecolor='black')
axes[1].axhline(y=overall, color='red', linestyle='--', lw=1.5)
axes[1].set_title('Fraud Rate by Day of Week', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Fraud Rate')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)

plt.tight_layout()
plt.savefig('eda_time.png', dpi=150, bbox_inches='tight')
plt.show()
""")

C13_eda_devices = code("""\
# Device-sharing analysis
device_user_counts_eda = df.groupby('device_id')['user_id'].nunique()
print("Device sharing distribution:")
print(device_user_counts_eda.value_counts().sort_index().rename('device_count').to_frame())

# Fraud rate for shared vs. unshared devices
df['_dev_users'] = df['device_id'].map(device_user_counts_eda)
shared_fraud = df.groupby(df['_dev_users'] > 1)['is_fraud'].agg(['mean', 'count'])
shared_fraud.index = ['Unshared device', 'Shared device (2+ users)']
print("\\nFraud rate by device sharing:")
print(shared_fraud.rename(columns={'mean': 'fraud_rate', 'count': 'n_txns'}).round(4))
df.drop(columns=['_dev_users'], inplace=True)
""")

C14_fe_header = md("### 2.3 Feature Engineering")

C15_fe_time = code("""\
# ── Time features ─────────────────────────────────────────────────────────────
# hour and day_of_week already created in EDA section

df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['is_night']         = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
df['is_business_hours']= (
    (df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)
).astype(int)

print("Time features:")
print(df[['timestamp', 'hour', 'day_of_week',
          'is_weekend', 'is_night', 'is_business_hours']].head(6))
""")

C16_fe_rolling = code("""\
# ── Rolling window features (mirrors the SQL query from Part 1) ────────────────
# In production these values would come directly from the SQL pipeline output.
# They are recomputed here in Python to serve as model training features.
#
# Semantics:
#   user_24h_amount         – total USD this user spent in the 24 h BEFORE this tx
#   user_tx_count_24h       – number of transactions this user made in last 24 h
#   user_1h_distinct_countries – distinct card_countries used in last 1 h

df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
print(f"Computing rolling features for {df['user_id'].nunique():,} users "
      f"over {len(df):,} transactions …")

# 24 h sum and count — pandas time-indexed rolling with closed='left'
# closed='left' = window [t-24h, t), i.e. strictly preceding the current row
rolling_24h_list = []
for _, group in df.groupby('user_id', sort=False):
    g = group.set_index('timestamp').sort_index()
    r = g['amount_usd'].rolling('24h', closed='left')
    rolling_24h_list.append(pd.DataFrame({
        'user_24h_amount'  : r.sum().fillna(0).values,
        'user_tx_count_24h': r.count().fillna(0).values,
    }, index=group.index))

rolling_24h_df = pd.concat(rolling_24h_list)
df['user_24h_amount']   = rolling_24h_df['user_24h_amount']
df['user_tx_count_24h'] = rolling_24h_df['user_tx_count_24h']

# 1 h distinct countries — pandas rolling doesn't support nunique() with time
# windows, so we use a NumPy per-group loop (avg group size ≈ 5.9 rows → fast).
countries_1h = np.zeros(len(df), dtype=np.int32)
for _, group in df.groupby('user_id', sort=False):
    g   = group.sort_values('timestamp')
    ts  = g['timestamp'].values
    cc  = g['card_country'].values
    idx = g.index.values
    for i in range(len(g)):
        window_start = ts[i] - np.timedelta64(1, 'h')
        mask = (ts < ts[i]) & (ts >= window_start)
        countries_1h[idx[i]] = len(set(cc[mask]))

df['user_1h_distinct_countries'] = countries_1h

print("\\nRolling window features — sample:")
print(df[['user_id', 'timestamp', 'amount_usd',
          'user_24h_amount', 'user_tx_count_24h',
          'user_1h_distinct_countries']].head(10).to_string(index=False))
""")

C17_fe_device = code("""\
# ── Device risk features ──────────────────────────────────────────────────────
device_user_count = (
    df.groupby('device_id')['user_id']
    .nunique()
    .rename('device_user_count')
    .reset_index()
)
df = df.merge(device_user_count, on='device_id', how='left')
df['device_shared']   = (df['device_user_count'] > 1).astype(int)
df['device_high_risk']= (df['device_user_count'] >= 3).astype(int)

print("Device feature stats:")
print(df.groupby('device_high_risk')['is_fraud']
      .agg(fraud_rate='mean', n='count').round(4))
""")

C18_fe_user = code("""\
# ── User behavioural features ─────────────────────────────────────────────────
user_stats = (
    df.groupby('user_id')
    .agg(
        user_mean_amount    =('amount_usd', 'mean'),
        user_std_amount     =('amount_usd', 'std'),
        user_total_txns     =('transaction_id', 'count'),
        user_unique_countries=('card_country', 'nunique'),
    )
    .reset_index()
)
user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)

df = df.merge(user_stats, on='user_id', how='left')

# Amount z-score relative to this user's personal baseline
df['amount_vs_user_mean'] = (
    (df['amount_usd'] - df['user_mean_amount'])
    / (df['user_std_amount'] + 1.0)   # +1 avoids division by zero for single-tx users
)

# Is this the first time this user has transacted from this card_country?
# Computed chronologically — no lookahead.
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
is_new_country = np.zeros(len(df), dtype=np.int32)
for _, group in df.groupby('user_id', sort=False):
    seen = set()
    for row_idx in group.index:            # group already in timestamp order
        country = df.at[row_idx, 'card_country']
        if country not in seen:
            is_new_country[row_idx] = 1
        seen.add(country)

df['is_new_country'] = is_new_country

print("User behavioural features — fraud rate breakdown:")
print(df.groupby('is_new_country')['is_fraud']
      .agg(fraud_rate='mean', n='count').round(4))
""")

C19_fe_matrix = code("""\
# ── Encode categoricals & assemble feature matrix ─────────────────────────────
# NOTE: dispute_status is deliberately EXCLUDED from FEATURE_COLS.
#       It will only be used in the post-model secondary validation layer.

le_country  = LabelEncoder()
le_category = LabelEncoder()
df['card_country_enc']       = le_country.fit_transform(df['card_country'])
df['merchant_category_enc']  = le_category.fit_transform(df['merchant_category'])

FEATURE_COLS = [
    # Time
    'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
    # Amount
    'amount_usd', 'amount_vs_user_mean',
    # Rolling window features (matches SQL Part 1)
    'user_24h_amount', 'user_tx_count_24h', 'user_1h_distinct_countries',
    # User profile
    'user_mean_amount', 'user_std_amount', 'user_total_txns', 'user_unique_countries',
    # Device
    'device_user_count', 'device_shared', 'device_high_risk',
    # New country flag
    'is_new_country',
    # Encoded categoricals
    'card_country_enc', 'merchant_category_enc',
]

X = df[FEATURE_COLS].copy()
y = df['is_fraud'].copy()

print(f"Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Fraud labels   : {y.sum():,} ({y.mean():.2%})")
print(f"\\nFeatures ({len(FEATURE_COLS)}):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")
""")

C20_model_header = md("### 2.4 Model Training — XGBoost Classifier")

C21_model_train = code("""\
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"Train : {len(X_train):,} rows  |  fraud rate = {y_train.mean():.2%}")
print(f"Test  : {len(X_test):,}  rows  |  fraud rate = {y_test.mean():.2%}")

# scale_pos_weight compensates for class imbalance without discarding majority data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\\nscale_pos_weight : {scale_pos_weight:.1f}  "
      f"(weights minority class {scale_pos_weight:.0f}× higher during training)")

model = xgb.XGBClassifier(
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    min_child_weight   = 5,
    subsample          = 0.80,
    colsample_bytree   = 0.80,
    gamma              = 1,
    reg_alpha          = 0.10,
    reg_lambda         = 1.0,
    scale_pos_weight   = scale_pos_weight,
    eval_metric        = 'aucpr',
    early_stopping_rounds = 40,
    random_state       = RANDOM_STATE,
    n_jobs             = -1,
)

model.fit(
    X_train, y_train,
    eval_set   = [(X_train, y_train), (X_test, y_test)],
    verbose    = 100,
)

print(f"\\nBest round       : {model.best_iteration}")
print(f"Best PR-AUC (val): {model.best_score:.4f}")
""")

C22_eval_header = md("### 2.5 Model Evaluation")

C23_eval_core = code("""\
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_05    = (y_pred_proba >= 0.50).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc  = average_precision_score(y_test, y_pred_proba)

print("=" * 55)
print("  MODEL PERFORMANCE  (XGBoost, threshold = 0.50)")
print("=" * 55)
print(f"  ROC-AUC            : {roc_auc:.4f}")
print(f"  PR-AUC             : {pr_auc:.4f}")
print(f"  (No-skill baseline : {y_test.mean():.4f})")
print("=" * 55)
print()
print(classification_report(y_test, y_pred_05,
      target_names=['Legitimate', 'Fraudulent'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_05)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['True Legit', 'True Fraud'])
ax.set_title(
    f'Confusion Matrix (threshold = 0.50)\\nROC-AUC = {roc_auc:.3f}  |  PR-AUC = {pr_auc:.3f}',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
""")

C24_eval_curves = code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='navy', lw=2.5, label=f'XGBoost  AUC = {roc_auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random classifier')
axes[0].fill_between(fpr, tpr, alpha=0.12, color='navy')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# ── Precision-Recall Curve ────────────────────────────────────────────────────
prec_vals, rec_vals, pr_thresh = precision_recall_curve(y_test, y_pred_proba)
baseline = y_test.mean()
axes[1].plot(rec_vals, prec_vals, color='crimson', lw=2.5,
             label=f'XGBoost  AP = {pr_auc:.3f}')
axes[1].axhline(y=baseline, color='gray', linestyle='--', lw=1.5,
                label=f'No-skill baseline ({baseline:.3f})')
axes[1].fill_between(rec_vals, prec_vals, baseline, alpha=0.12, color='crimson')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title(
    'Precision-Recall Curve\\n(primary metric for imbalanced fraud detection)',
    fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("PR-AUC is the primary metric because:")
print("  • The class imbalance (98:2) makes ROC-AUC optimistic.")
print("  • PR-AUC directly measures how well the model identifies the minority fraud class.")
print(f"  • A random classifier achieves PR-AUC ≈ {baseline:.3f}; model achieves {pr_auc:.3f}.")
""")

C25_threshold = code("""\
thresholds = np.arange(0.05, 0.96, 0.05)
rows = []
for t in thresholds:
    pred = (y_pred_proba >= t).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    rows.append({
        'threshold': round(float(t), 2),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall'   : recall_score(y_test, pred, zero_division=0),
        'f1'       : f1_score(y_test, pred, zero_division=0),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
    })

thresh_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(thresh_df['threshold'], thresh_df['precision'], 'b-o', ms=4, label='Precision')
axes[0].plot(thresh_df['threshold'], thresh_df['recall'],    'r-o', ms=4, label='Recall')
axes[0].plot(thresh_df['threshold'], thresh_df['f1'],        'g-o', ms=4, label='F1 Score')
axes[0].axvline(x=0.50, color='gray', linestyle='--', alpha=0.7, label='Default (0.50)')
axes[0].set_xlabel('Decision Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Precision / Recall / F1 vs. Threshold', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresh_df['threshold'], thresh_df['tp'], 'g-o', ms=4,
             label='True Positives  (fraud caught)')
axes[1].plot(thresh_df['threshold'], thresh_df['fp'], 'r-o', ms=4,
             label='False Positives (legit blocked)')
axes[1].plot(thresh_df['threshold'], thresh_df['fn'], color='orange', marker='o', ms=4,
             label='False Negatives (fraud missed)')
axes[1].set_xlabel('Decision Threshold')
axes[1].set_ylabel('Count')
axes[1].set_title('TP / FP / FN Counts vs. Threshold', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

best_f1_row = thresh_df.loc[thresh_df['f1'].idxmax()]
print(f"\\nOptimal F1 threshold : {best_f1_row['threshold']:.2f}")
print(f"  Precision = {best_f1_row['precision']:.3f}")
print(f"  Recall    = {best_f1_row['recall']:.3f}")
print(f"  F1        = {best_f1_row['f1']:.3f}")
print(f"  TP={int(best_f1_row['tp'])}, FP={int(best_f1_row['fp'])}, FN={int(best_f1_row['fn'])}")
print()
print(thresh_df[['threshold','precision','recall','f1','tp','fp','fn','fpr']].to_string(index=False))
""")

C26_feat_imp = code("""\
fi_df = (
    pd.DataFrame({'feature': FEATURE_COLS,
                  'importance': model.feature_importances_})
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)

median_imp = fi_df['importance'].median()
colors = ['#d62728' if v > median_imp else '#1f77b4' for v in fi_df['importance']]

plt.figure(figsize=(10, 7))
plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1],
         color=colors[::-1], alpha=0.85, edgecolor='black', linewidth=0.5)
plt.xlabel('Feature Importance (gain)', fontsize=11)
plt.title('XGBoost Feature Importance\\n(red = above-median importance)',
          fontsize=13, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 10 features:")
print(fi_df.head(10).to_string(index=False))
""")

C27_secondary_header = md("""\
### 2.6 Secondary Validation Layer — `dispute_status` Override (Option C)

Because `dispute_status` cannot exist at scoring time for new transactions, the model was
trained **without** it.  However, it *is* available for:

- **Retrospective monitoring** – reviewing transactions after the fact
- **Review queue prioritisation** – surfacing the highest-risk cases for analysts

The secondary validation layer applies a score **override** after the model scores a
transaction, without retraining or altering the model weights.

| Status | Action | Rationale |
|---|---|---|
| `chargeback_won` | Force score → **1.0** | Confirmed fraud — already resolved |
| `pending` | Force score → **1.0** | Confirmed fraud — under chargeback process |
| `inquiry` | Boost score by **+0.10** | Issuer flagged for review; not confirmed |
| `none` | Use model score unchanged | No external signal |
""")

C28_secondary_code = code("""\
def apply_secondary_validation(model_score: float, dispute_status: str) -> float:
    \"\"\"
    Post-model dispute_status override.
    Does NOT alter model weights — purely a decision-layer adjustment.
    \"\"\"
    if dispute_status in ('chargeback_won', 'pending'):
        return 1.0
    elif dispute_status == 'inquiry':
        return min(1.0, model_score + 0.10)
    return model_score


test_df = df.loc[X_test.index, ['is_fraud', 'dispute_status']].copy()
test_df['model_score'] = y_pred_proba
test_df['final_score'] = test_df.apply(
    lambda r: apply_secondary_validation(r['model_score'], r['dispute_status']),
    axis=1
)
test_df['final_pred'] = (test_df['final_score'] >= 0.50).astype(int)

final_pr_auc  = average_precision_score(test_df['is_fraud'], test_df['final_score'])
final_roc_auc = roc_auc_score(test_df['is_fraud'], test_df['final_score'])

print("=" * 62)
print("  COMPARISON: Model-only  vs.  Model + Secondary Validation")
print("=" * 62)

print("\\n── Model only (threshold = 0.50) ──")
print(classification_report(y_test, y_pred_05,
      target_names=['Legitimate', 'Fraudulent'], digits=4))

print("── Model + dispute_status override (threshold = 0.50) ──")
print(classification_report(test_df['is_fraud'], test_df['final_pred'],
      target_names=['Legitimate', 'Fraudulent'], digits=4))

print(f"Model only     →  ROC-AUC = {roc_auc:.4f}  |  PR-AUC = {pr_auc:.4f}")
print(f"With override  →  ROC-AUC = {final_roc_auc:.4f}  |  PR-AUC = {final_pr_auc:.4f}")

# How many cases the override changed
overridden = (test_df['final_score'] != test_df['model_score']).sum()
print(f"\\nRows affected by override : {overridden:,} "
      f"({overridden/len(test_df):.1%} of test set)")
""")

C29_summary_cell = md("""\
---
## Results Summary

| Metric | Value |
|---|---|
| Dataset | 52,500 transactions, 1.9% fraud rate |
| Features used | 20 (time, amount, rolling-window, device, user-behavioural) |
| Algorithm | XGBoost with `scale_pos_weight` for class imbalance |
| Evaluation split | 80% train / 20% test, stratified |

> The model deliberately excludes `dispute_status` to avoid training-time leakage.
> The secondary validation layer adds it back as a post-model operational override.

**See `business_strategy_summary.md` for the full Part 3 business strategy response.**
""")

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE & WRITE
# ─────────────────────────────────────────────────────────────────────────────

nb.cells = [
    C00_title,
    C01_setup,
    C02_part1_header,
    C03_sql,
    C04_sql_perf,
    C05_part2_header,
    C06_load,
    C07_timestamps,
    C08_eda_fraud_dist,
    C09_eda_class,
    C10_leakage,
    C11_eda_cat,
    C12_eda_time,
    C13_eda_devices,
    C14_fe_header,
    C15_fe_time,
    C16_fe_rolling,
    C17_fe_device,
    C18_fe_user,
    C19_fe_matrix,
    C20_model_header,
    C21_model_train,
    C22_eval_header,
    C23_eval_core,
    C24_eval_curves,
    C25_threshold,
    C26_feat_imp,
    C27_secondary_header,
    C28_secondary_code,
    C29_summary_cell,
]

OUTPUT = 'fraud_detection.ipynb'
with open(OUTPUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✓  Written: {OUTPUT}  ({len(nb.cells)} cells)")
