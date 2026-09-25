from nbtools import NB

nb = NB("NB3_cleaning_transformation_pipelines.ipynb")
nb.header("Notebook 3 – Cleaning, transformation and pipelines",
          "3",
          "Clean the churn extract, deal with structural missingness and outliers, engineer features relative to the "
          "prediction point, and put every learned step in one scikit-learn pipeline that is fitted on the training data only.",
          25)

nb.md("""
We continue with the churn case of Notebook 1 (prediction point 2023-06-01, churn during June).
The leaky columns identified in Notebook 1 are removed right away.

Two kinds of preprocessing steps:

| Kind | Examples | Where |
|---|---|---|
| **Rules** that do not learn anything from the data | fix spellings, parse dates, flag impossible values, compute tenure | may run before the split, but must also run in production |
| **Learned** steps | imputation values, scaling parameters, encodings, selected features | inside a pipeline, fitted on the training data only |
""")
nb.data_cell("skrub")
nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

set_config(transform_output="pandas")      # transformers return DataFrames with readable column names
pd.set_option("display.width", 200)
PREDICTION_POINT = pd.Timestamp("2023-06-01")

raw = pd.read_csv(DATA + "churn.csv", dtype={"region_code": str})
raw = raw.drop(columns=["minutes_used_current_month", "last_call_date", "helpdesk_calls_3m"])   # leaky, see Notebook 1
print(raw.shape)
""")

nb.md("""
## 3.1 Basic consistency
""")
nb.code("""
print(raw["contract_type"].value_counts(), "\\n")
print(raw["region_code"][~raw["region_code"].str.fullmatch(r"\\d{4}")].value_counts())
print("\\nDates not in ISO format:", (~raw["contract_start_date"].str.fullmatch(r"\\d{4}-\\d{2}-\\d{2}")).sum())
print("Exact duplicates:", raw.duplicated().sum())
""")
nb.code("""
def clean(df):
    \"\"\"Deterministic cleaning rules. Nothing is learned from the data, so the same function runs in production.\"\"\"
    df = df.drop_duplicates().copy()
    ct = df["contract_type"].str.lower().str.replace("-", "", regex=False)
    df["contract_type"] = np.where(ct.str.startswith("pre"), "Prepaid", "Postpaid")
    df["postal_code"] = df["region_code"].str.extract(r"(\\d{4})", expand=False)
    iso = pd.to_datetime(df["contract_start_date"], format="%Y-%m-%d", errors="coerce")
    df["contract_start_date"] = iso.fillna(pd.to_datetime(df["contract_start_date"], format="%d/%m/%Y", errors="coerce"))
    return df.drop(columns=["region_code"])

churn = clean(raw)
print(churn["contract_type"].value_counts())
churn.head()
""")

nb.md("""
## 3.2 Split before anything is learned
""")
nb.code("""
train, test = train_test_split(churn, test_size=0.3, stratify=churn["churn"], random_state=42)
y_train, y_test = train["churn"], test["churn"]
print(train.shape, test.shape)
""")

nb.md("""
## 3.3 Missing values: why are they missing?
""")
nb.code("""
print(train.isna().mean().round(3)[lambda s: s > 0])
train.assign(bill_missing=train["avg_monthly_bill"].isna()).groupby("contract_type")["bill_missing"].mean().round(3)
""")
nb.md("""
Prepaid customers receive no monthly bill: for them the value is **structurally** missing (not applicable).
We set their bill to 0 (the contract type already tells the model they are prepaid).
The few remaining gaps for postpaid customers look random; they are imputed with the **training** median inside the pipeline,
together with a missing-indicator column.
""")

nb.md("""
## 3.4 Outliers: invalid or valid?
""")
nb.code("""
train[["minutes_used_two_months_ago", "minutes_used_last_month", "avg_monthly_bill", "unpaid_invoices"]].describe(
    percentiles=[0.01, 0.5, 0.99]).round(1)
""")
nb.md("""
- Negative minutes, 500 unpaid invoices and a bill of 99,999 EUR are **impossible**: treat them as missing.
- 3,000+ minutes a month is rare but **possible** (heavy users). Keep them. For a linear model, a log transform
  reduces their influence; tree-based models are hardly affected.
""")

nb.md("""
## 3.5 Feature engineering relative to the prediction point

Row-wise features learn nothing from the data, so they can be computed with a function that we apply to train, test
and later production data alike.
""")
nb.code("""
def make_features(df):
    out = pd.DataFrame(index=df.index)
    out["contract_type"] = df["contract_type"]
    out["postal_code"] = df["postal_code"]
    out["postal_area"] = df["postal_code"].str[:2]                      # grouping: 280 codes -> ~ 80 areas
    out["tenure_days"] = (PREDICTION_POINT - df["contract_start_date"]).dt.days
    out["is_new_customer"] = (out["tenure_days"] < 180).astype(int)
    minutes_apr = df["minutes_used_two_months_ago"].where(df["minutes_used_two_months_ago"] >= 0)   # invalid -> NaN
    minutes_may = df["minutes_used_last_month"].where(df["minutes_used_last_month"] >= 0)
    out["minutes_april"], out["minutes_may"] = minutes_apr, minutes_may
    out["usage_ratio"] = minutes_may / (minutes_apr + 1)                 # trend: May vs April
    out["usage_drop"] = (out["usage_ratio"] < 0.8).astype(int)
    bill = df["avg_monthly_bill"].where(df["avg_monthly_bill"] < 2000)   # invalid -> NaN
    out["avg_monthly_bill"] = bill.mask(df["contract_type"] == "Prepaid", 0.0)   # structural: no bill for prepaid
    out["unpaid_invoices"] = df["unpaid_invoices"].where(df["unpaid_invoices"] < 50)
    return out

X_train, X_test = make_features(train), make_features(test)
X_train.head()
""")

nb.md("""
## 3.6 Learned transformations in one ColumnTransformer

- numeric: impute (median, plus missing indicator) → log (for skewed features) → standardise
- `contract_type`: one-hot encoding
- `postal_code` (high cardinality): target encoding. `TargetEncoder` uses internal cross-fitting, so a row's own
  label is never used to encode that row.
""")
nb.code("""
NUMERIC = ["tenure_days", "usage_ratio", "avg_monthly_bill", "unpaid_invoices"]
SKEWED = ["minutes_april", "minutes_may"]
BINARY = ["is_new_customer", "usage_drop"]

def build_pipeline(impute="median", add_indicator=True, use_log=True, region="target", model="logreg"):
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one") if use_log else "passthrough"
    steps = [
        ("num", make_pipeline(SimpleImputer(strategy=impute, add_indicator=add_indicator), StandardScaler()), NUMERIC),
        ("skewed", make_pipeline(SimpleImputer(strategy=impute), log, StandardScaler()), SKEWED),
        ("contract", OneHotEncoder(drop="if_binary", sparse_output=False), ["contract_type"]),
        ("binary", "passthrough", BINARY),
    ]
    if region == "target":
        steps.append(("region", TargetEncoder(random_state=0), ["postal_code"]))
    elif region == "onehot":
        steps.append(("region", OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist", sparse_output=False), ["postal_code"]))
    preprocess = ColumnTransformer(steps, verbose_feature_names_out=False)
    clf = LogisticRegression(max_iter=2000) if model == "logreg" else HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0)
    return make_pipeline(preprocess, clf)

pipe = build_pipeline()
pipe[0].fit(X_train, y_train)
pipe[0].transform(X_train).head()
""")

nb.md("""
## 3.7 Fit the whole pipeline on the training data, evaluate on the test data
""")
nb.code("""
def evaluate(pipe, label):
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    test = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    return pd.Series({"CV AUC (train)": round(cv, 3), "test AUC": round(test, 3)}, name=label)

results = [evaluate(build_pipeline(), "logreg + target encoding")]
pd.DataFrame(results)
""")
nb.md("""
The pipeline object *is* the model: it contains the medians, scaling parameters and encodings learned on the training
data. In production you call `pipe.predict_proba(new_customers)` and the exact same steps are applied.
""")

nb.md("""
## 3.8 Leakage through preprocessing

A common mistake: encode the postal code with the average churn per postal code computed on the **whole** dataset,
before splitting. The test rows then contribute their own label to their own feature.
""")
nb.code("""
all_rows = pd.concat([X_train.assign(churn=y_train), X_test.assign(churn=y_test)])
naive_map = all_rows.groupby("postal_code")["churn"].mean()          # computed on train + test: wrong
Xn_train = X_train.assign(postal_churn=X_train["postal_code"].map(naive_map))
Xn_test = X_test.assign(postal_churn=X_test["postal_code"].map(naive_map))

naive = build_pipeline(region="drop")
naive.steps[0][1].transformers.append(("postal_churn", "passthrough", ["postal_churn"]))
naive.fit(Xn_train, y_train)
leaky_auc = roc_auc_score(y_test, naive.predict_proba(Xn_test)[:, 1])
print("Test AUC with naive target encoding (leaky):", round(leaky_auc, 3))
print("Test AUC with TargetEncoder inside the pipeline:", results[0]["test AUC"])
""")
nb.md("""
The leaky version looks better on the test set, but that advantage disappears in production, where the churn label of a
new customer is unknown. Rare postal codes are the worst: with 3 customers, their "average churn" is mostly their own label.
""")

nb.md("""
## 3.9 How much preprocessing do modern models need?

Gradient boosting (`HistGradientBoostingClassifier`) handles missing values and categorical features natively and
does not need scaling. `skrub.tabular_pipeline` builds a strong default pipeline for a raw dataframe in one line.
""")
nb.code("""
hgb_cols = ["contract_type", "postal_area", "tenure_days", "minutes_april", "minutes_may", "usage_ratio",
            "avg_monthly_bill", "unpaid_invoices"]
def to_category(X):
    return X[hgb_cols].astype({"contract_type": "category", "postal_area": "category"})

hgb = make_pipeline(FunctionTransformer(to_category),
                    HistGradientBoostingClassifier(categorical_features="from_dtype", max_depth=3,
                                                   learning_rate=0.05, random_state=0))
results.append(evaluate(hgb, "gradient boosting, no imputation or scaling"))

import skrub
baseline = skrub.tabular_pipeline("classifier") if hasattr(skrub, "tabular_pipeline") else skrub.tabular_learner("classifier")
raw_train = train.drop(columns=["churn", "customer_id"])
raw_test = test.drop(columns=["churn", "customer_id"])
cv = cross_val_score(baseline, raw_train, y_train, cv=5, scoring="roc_auc").mean()
baseline.fit(raw_train, y_train)
results.append(pd.Series({"CV AUC (train)": round(cv, 3),
                          "test AUC": round(roc_auc_score(y_test, baseline.predict_proba(raw_test)[:, 1]), 3)},
                         name="skrub baseline on cleaned raw columns"))
pd.DataFrame(results)
""")
nb.md("""
Notice what the automated baseline cannot do on its own: it does not know the prediction point (it sees a start *date*,
not a tenure), and it would happily use leaky columns if we had left them in. Mechanical preprocessing is increasingly
automated; defining the prediction point, the target and the allowed information is not.
""")

nb.exercise("3A – Compare preprocessing choices (run & tweak)", """
Change the options and rerun. Keep a small table of your results.

1. Does the missing-indicator help? Does the imputation strategy matter here?
2. What happens without the log transform for the logistic regression? And for gradient boosting (`model="hgb"`)?
3. Compare `region="target"`, `"onehot"` and `"drop"`. Is the postal code worth its complexity?
""")
nb.code("""
options = dict(impute="median",        # <- change: "median", "mean", "most_frequent"
               add_indicator=True,     # <- change
               use_log=True,           # <- change
               region="target",        # <- change: "target", "onehot", "drop"
               model="logreg")         # <- change: "logreg", "hgb"
evaluate(build_pipeline(**options), str(options)).to_frame().T
""")
nb.solution("Solution – Exercise 3A", """
rows = []
for opts in [dict(), dict(add_indicator=False), dict(impute="mean"), dict(use_log=False),
             dict(region="onehot"), dict(region="drop"), dict(model="hgb"), dict(model="hgb", use_log=False)]:
    rows.append(evaluate(build_pipeline(**opts), str(opts) if opts else "default"))
print(pd.DataFrame(rows))
# Typically: imputation choices barely matter (few gaps); the log transform matters for logistic regression but not
# for boosting; the postal code adds a little signal, target encoding keeps it to one column.
""")

nb.exercise("3B – Write your own learned transformer (write code)", """
Write a `Winsorizer` that caps every column at the 1st and 99th percentile **learned during `fit`** (so on the
training data only), and use it instead of the log transform. Hint: subclass `BaseEstimator` and `TransformerMixin`,
store the percentiles in `fit`, apply `clip` in `transform`.

Extra: add an engineered feature of your own to `make_features` and check whether it beats the skrub baseline.
""")
nb.code("""
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    # your code here
    pass
""")
nb.solution("Solution – Exercise 3B", """
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_, self.high_ = X.quantile(self.lower), X.quantile(self.upper)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self
    def transform(self, X):
        return pd.DataFrame(X).clip(self.low_, self.high_, axis=1)
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

pipe_w = build_pipeline(use_log=False)
pipe_w[0].transformers[1] = ("skewed", make_pipeline(SimpleImputer(strategy="median"), Winsorizer(), StandardScaler()), SKEWED)
print(evaluate(pipe_w, "logreg + winsorizing"))
""")

nb.md("""
## 3.10 Take-aways

- Separate **rules** (deterministic, reusable in production) from **learned** steps (inside the pipeline, fitted on train).
- Ask *why* a value is missing or extreme before choosing a treatment; extremes can be the signal.
- The pipeline is the model: deploy it as one object.

**Back to the slides: part 4, Anomaly detection.**
""")

nb.save("../notebooks")
