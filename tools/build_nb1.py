from nbtools import NB

nb = NB("NB1_selection_and_leakage.ipynb")
nb.header("Notebook 1 – Data selection, the prediction point and leakage",
          "1",
          "Turn a raw churn extract into a modelling dataset, set aside a hold-out set, and see how "
          "features that are not available at prediction time make a model look far better than it is.",
          10)

nb.md("""
## The case

A telecom operator wants to predict churn. We use the dataset from the in-class exercise:

| | |
|---|---|
| Export date of the data | **2023-07-01** |
| Prediction point | **2023-06-01** (we predict on this day) |
| Churn window | **June 2023** (target = churned during June) |

Every feature must describe the world **as it was on June 1**. Anything that happened in June belongs to the outcome, not to the inputs.
""")

nb.data_cell()
nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
PREDICTION_POINT = pd.Timestamp("2023-06-01")
EXPORT_DATE = pd.Timestamp("2023-07-01")
""")

nb.md("""
## 1.1 Look at the raw excerpt first

This is the 36-row excerpt you may know from the exercise. Before any code, just look at it.
Which columns worry you? Which values look odd?
""")
nb.code("""
sample = pd.read_excel(DATA + "churn_sample.xlsx")
sample.head(15)
""")

nb.md("""
## 1.2 The full extract

For the hands-on part we use a larger extract (5,040 rows) with the same columns, generated to behave like the excerpt.
It adds one column, `minutes_used_two_months_ago` (April usage), so that we can compute usage trends later.
""")
nb.code("""
churn = pd.read_csv(DATA + "churn.csv")
print(churn.shape)
print("Churn rate:", round(churn["churn"].mean(), 3))
churn.head()
""")

nb.md("""
## 1.3 Where is each column in time?

A quick check on the dates tells a lot. If `last_call_date` contains June dates, the column was filled in *after* the prediction point.
""")
nb.code("""
last_call = pd.to_datetime(churn["last_call_date"])
print("Last call dates range from", last_call.min().date(), "to", last_call.max().date())
print("Share of customers with a last call AFTER the prediction point:",
      round((last_call >= PREDICTION_POINT).mean(), 3))

# Who are they? Compare churners and non-churners
print(churn.assign(call_after_pp=last_call >= PREDICTION_POINT)
           .groupby("churn")["call_after_pp"].mean().round(3))
""")
nb.md("""
Active customers keep calling in June, churners do not. The column therefore "knows" the outcome.
The same reasoning applies to `minutes_used_current_month` (June usage) and to `helpdesk_calls_3m`,
whose three-month window ends at the export date and therefore includes June.
""")

nb.md("""
## 1.4 Set aside a hold-out set before anything is learned

Rule: split **before** you compute anything from the data (means for imputation, scaling parameters, encodings, selected features).
Duplicated rows would end up in both train and test (instance leakage), so we remove exact duplicates first.
""")
nb.code("""
print("Exact duplicate rows:", churn.duplicated().sum())
churn = churn.drop_duplicates().reset_index(drop=True)

train, test = train_test_split(churn, test_size=0.3, stratify=churn["churn"], random_state=42)
print("Train:", train.shape, " Test:", test.shape)
""")
nb.md("""
*Note.* In a real churn project you would rather use a **time-based** split: build the training set from an earlier
snapshot (e.g. prediction point May 1, churn in May) and test on the June snapshot. With a single snapshot, a stratified
random split is the best we can do.
""")

nb.md("""
## 1.5 Leakage makes a model look (too) good

A very small feature preparation, just enough to train a model. Cleaning is the topic of Notebook 3, so do not worry about the details here.
""")
nb.code("""
def quick_features(df):
    out = pd.DataFrame(index=df.index)
    out["prepaid"] = df["contract_type"].str.lower().str.startswith("pre").astype(int)
    iso = pd.to_datetime(df["contract_start_date"], format="%Y-%m-%d", errors="coerce")
    start = iso.fillna(pd.to_datetime(df["contract_start_date"], format="%d/%m/%Y", errors="coerce"))
    out["tenure_days"] = (PREDICTION_POINT - start).dt.days
    out["minutes_april"] = df["minutes_used_two_months_ago"]
    out["minutes_may"] = df["minutes_used_last_month"]
    out["usage_ratio"] = df["minutes_used_last_month"] / (df["minutes_used_two_months_ago"] + 1)
    out["unpaid_invoices"] = df["unpaid_invoices"]
    out["avg_monthly_bill"] = df["avg_monthly_bill"]
    # suspicious candidates
    out["minutes_june"] = df["minutes_used_current_month"]
    out["days_since_last_call"] = (EXPORT_DATE - pd.to_datetime(df["last_call_date"])).dt.days
    out["helpdesk_calls_3m"] = df["helpdesk_calls_3m"]
    return out

X_train, X_test = quick_features(train), quick_features(test)
y_train, y_test = train["churn"], test["churn"]

def test_auc(features):
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(X_train[features], y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test[features])[:, 1])

SAFE = ["prepaid", "tenure_days", "minutes_april", "minutes_may", "usage_ratio", "unpaid_invoices", "avg_monthly_bill"]
results = {"safe features only": test_auc(SAFE)}
for extra in ["helpdesk_calls_3m", "days_since_last_call", "minutes_june"]:
    results[f"safe + {extra}"] = test_auc(SAFE + [extra])
pd.Series(results).round(3).to_frame("test AUC")
""")
nb.md("""
An AUC close to 1.0 is not a success, it is a warning sign: *too good to be true*.
The model would never reach this performance on June 1, because June usage and June calls do not exist yet on that day.
""")

nb.exercise("1A – Which features are allowed? (run & tweak)", """
Edit the list `MY_FEATURES` below: add or remove features and rerun the cell.

1. Which feature on its own pushes the AUC closest to 1? Why?
2. `helpdesk_calls_3m` gives a smaller jump. Is it still leakage? What would you ask the data owner?
3. Write down the final list of features you would allow at the prediction point.
""")
nb.code("""
MY_FEATURES = SAFE + ["helpdesk_calls_3m"]      # <- change
print("Test AUC:", round(test_auc(MY_FEATURES), 3))
""")
nb.solution("Solution – Exercise 1A", """
# 1. minutes_june: it is measured during the churn window; churners stop using the service (AUC ~ 1).
# 2. Yes, partly: the 3-month window ends at the export date (April-June) and churners call to cancel in June.
#    Ask for the value computed on March-May, i.e. relative to the prediction point.
# 3. Allowed: the SAFE list. days_since_last_call and minutes_june are excluded; helpdesk_calls_3m only if recomputed.
for f in [SAFE, SAFE + ["helpdesk_calls_3m"], SAFE + ["minutes_june"]]:
    print(len(f), "features ->", round(test_auc(f), 3))
""")

nb.exercise("1B – A simple leakage alarm (optional, code)", """
Write a function `single_feature_auc(X, y, feature)` that returns the AUC of a model trained on **one** feature only.
Apply it to every column of `X_train` (use cross-validation on the training set, not the test set!) and flag features with an AUC above 0.9.

Second question: compute `tenure_days` relative to the **export date** instead of the prediction point. Does the model notice? Why does it still matter in deployment?
""")
nb.code("""
from sklearn.model_selection import cross_val_score

def single_feature_auc(X, y, feature):
    # your code here
    pass
""")
nb.solution("Solution – Exercise 1B", """
from sklearn.model_selection import cross_val_score

def single_feature_auc(X, y, feature):
    model = HistGradientBoostingClassifier(max_depth=2, random_state=0)
    return cross_val_score(model, X[[feature]], y, cv=5, scoring="roc_auc").mean()

alarm = pd.Series({f: single_feature_auc(X_train, y_train, f) for f in X_train.columns}).sort_values(ascending=False)
print(alarm.round(3))
print("Flagged:", list(alarm[alarm > 0.9].index))

# Tenure relative to the export date is a constant shift of 30 days: the model does not care,
# but in production the feature must be computed with the same reference date as in training,
# otherwise every customer looks one month older (training/serving skew).
""")

nb.md("""
## 1.6 Take-aways

- Define the **prediction point** first; every feature must be computable on that day.
- Split **before** anything is learned from the data, and remove duplicates first.
- A suspiciously high performance calls for investigation, not celebration.

**Back to the slides: part 2, Exploratory data analysis.**
""")

nb.save("../notebooks")
