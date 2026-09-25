from nbtools import NB

nb = NB("NB1_selection_and_leakage.ipynb")
nb.header("Notebook 1 – Data selection, the prediction point and leakage",
          "1",
          "Turn a raw churn extract into a modelling dataset, set aside a hold-out set, see how one leaky feature "
          "makes a model look far better than it is, and then act as a leakage detective on the remaining columns.",
          10)

nb.md("""
## The case

A telecom operator wants to predict churn. The data owner gives you this information:

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
PREDICTION_POINT = pd.Timestamp("2023-06-01")
EXPORT_DATE = pd.Timestamp("2023-07-01")
""")

nb.md("""
## 1.1 Look at the raw excerpt first

This is a 36-row excerpt of the data, as a data owner might e-mail it to you. Before any code, just look at it.
Which columns worry you? Which values look odd?
""")
nb.code("""
sample = pd.read_excel(DATA + "churn_sample.xlsx")
sample.head(15)
""")

nb.md("""
## 1.2 The full extract

For the hands-on part we use the full extract: 5,040 customers. The data owner added a few columns to the excerpt.

| Column | Description (as given by the data owner) |
|---|---|
| `customer_id` | Unique identifier |
| `contract_type` | Prepaid / Postpaid |
| `contract_start_date` | Date the contract started |
| `last_call_date` | Date of the last call |
| `minutes_used_two_months_ago` | Minutes used in April |
| `minutes_used_last_month` | Minutes used in May |
| `minutes_used_current_month` | Minutes used in the current month |
| `helpdesk_calls_3m` | Number of helpdesk calls in the last 3 months |
| `avg_monthly_bill` | Average monthly bill (EUR) |
| `unpaid_invoices` | Number of unpaid invoices |
| `region_code` | Postal code |
| `retention_offer_sent` | Customer received a retention offer (1/0) |
| `region_churn_rate` | Average churn rate in the customer's postal code |
| `contract_end_date` | End date of the contract |
| `churn` | Target: churned in June (1/0) |
""")
nb.code("""
churn = pd.read_csv(DATA + "churn.csv")
print(churn.shape)
print("Churn rate:", round(churn["churn"].mean(), 3))
churn.head()
""")

nb.md("""
## 1.3 Where is each column in time?

The key question for every column: **was this value known on June 1?**
Take `minutes_used_current_month`. The data was exported on July 1, so the "current month" is June: the churn window itself.
""")
nb.code("""
churn.groupby("churn")[["minutes_used_last_month", "minutes_used_current_month"]].median()
""")
nb.md("""
Churners and non-churners use about the same number of minutes in May, but in June churners stop using the service.
The column describes the outcome, not the situation on June 1.
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
y_train, y_test = train["churn"], test["churn"]
print("Train:", train.shape, " Test:", test.shape)
""")
nb.md("""
*Note.* In a real churn project you would rather use a **time-based** split: build the training set from an earlier
snapshot (e.g. prediction point May 1, churn in May) and test on the June snapshot. With a single snapshot, a stratified
random split is the best we can do.
""")

nb.md("""
## 1.5 One leaky feature makes the model look (too) good

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
    out["minutes_june"] = df["minutes_used_current_month"]
    return out

X_train, X_test = quick_features(train), quick_features(test)

def test_auc(features, Xtr=None, Xte=None):
    Xtr = X_train if Xtr is None else Xtr
    Xte = X_test if Xte is None else Xte
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(Xtr[features], y_train)
    return roc_auc_score(y_test, model.predict_proba(Xte[features])[:, 1])

BASE = ["prepaid", "tenure_days", "minutes_april", "minutes_may", "usage_ratio", "unpaid_invoices", "avg_monthly_bill"]
pd.Series({"base features": test_auc(BASE),
           "base + minutes_june": test_auc(BASE + ["minutes_june"])}).round(3).to_frame("test AUC")
""")
nb.md("""
An AUC of 1.0 is not a success, it is a warning sign: *too good to be true*.
The model would never reach this performance on June 1, because June usage does not exist yet on that day.

`minutes_used_current_month` was easy to spot. Are there other leaky columns? Some are much harder to see.
""")

nb.md("""
## 1.6 Tools for a leakage detective

Two simple screening tools, applied to the **training data only**:

- `leakage_scan`: the cross-validated AUC of a small model that uses **one raw column at a time**.
  A column that predicts churn almost on its own deserves a closer look. Dates are converted to "days after the prediction point";
  a missing date is kept as information.
- `date_check`: for every date column, the share of values **after the prediction point**, and the share of missing values,
  for churners and non-churners.

A flag is not proof: a strong feature can be perfectly legitimate. Each flag needs an explanation: *when* was the value
recorded, and *from what* was it computed?
""")
nb.code("""
DATE_COLUMNS = ["contract_start_date", "last_call_date", "contract_end_date"]

def parse_dates(s):
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return iso.fillna(pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"))

def as_numeric(col):
    s = train[col]
    if col in DATE_COLUMNS:
        return (parse_dates(s) - PREDICTION_POINT).dt.days      # NaN (missing date) is kept
    if not pd.api.types.is_numeric_dtype(s):
        return s.astype("category").cat.codes                   # categories -> integer codes
    return s

def leakage_scan(columns, cv=5):
    model = HistGradientBoostingClassifier(max_depth=2, random_state=0)
    auc = {c: cross_val_score(model, as_numeric(c).to_frame(), y_train, cv=cv, scoring="roc_auc").mean() for c in columns}
    return pd.Series(auc).sort_values(ascending=False).round(3).to_frame("single-column AUC (train, CV)")

def date_check():
    rows = {}
    for c in DATE_COLUMNS:
        d = parse_dates(train[c])
        for label, grp in [("churners", y_train == 1), ("non-churners", y_train == 0)]:
            rows[(c, label)] = {"after prediction point": (d[grp] >= PREDICTION_POINT).mean(),
                                "missing": d[grp].isna().mean()}
    return pd.DataFrame(rows).T.round(3)

RAW_COLUMNS = [c for c in train.columns if c != "churn"]
print(len(RAW_COLUMNS), "columns to investigate")
""")

nb.exercise("1A – Leakage detective (run & tweak)", """
Besides `minutes_used_current_month`, **at least four other columns** leak information about the outcome.

1. Run `leakage_scan` and `date_check` below. Change `THRESHOLD` and look at the columns above and just below it.
2. For every flagged column, decide: **leakage or a legitimately strong feature?** Use the data dictionary in §1.2 and ask:
   *was this value known on June 1, and was it computed from information that is only available later?*
3. Not every leak has a high AUC, and not every high AUC is a leak. Which column surprised you?

Write your verdict in the table in the text cell below (double-click to edit).
""")
nb.code("""
THRESHOLD = 0.75          # <- change
scan = leakage_scan(RAW_COLUMNS)
display(scan.style.apply(lambda s: ["font-weight: bold" if v > THRESHOLD else "" for v in s]))
date_check()
""")
nb.md("""
| Column | Leak? | Why (when was it recorded, computed from what?) |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
""")
nb.solution("Solution – Exercise 1A", """
# Leaks (not known on June 1, or computed with information from after June 1):
# - minutes_used_current_month: June usage = the churn window itself.
# - last_call_date: extracted on July 1. Active customers keep calling in June, churners do not
#   (date_check: most non-churners have a last call AFTER the prediction point).
# - contract_end_date: filled in when a postpaid contract is terminated, i.e. in June for churners
#   (date_check: all non-missing end dates of churners fall after June 1). Its MISSINGNESS leaks too.
# - retention_offer_sent: the call centre makes an offer when a customer calls to cancel, in June.
#   Nothing in the name suggests a date: you have to ask the data owner when the flag is set.
# - region_churn_rate: computed on ALL customers, including each customer's own label and the test set.
#   For rare postal codes it is almost the customer's own label (a leak through preprocessing).
#   Its single-column AUC (~0.67) is no higher than that of legitimate features: the scan does not catch it,
#   reading the data dictionary ("average churn rate") does.
# - helpdesk_calls_3m: a milder leak. The 3-month window ends at the export date (April-June) and
#   churners call to cancel in June. Ask for the value on March-May.
#
# Not a leak: minutes_used_last_month (May) scores about as high as helpdesk_calls_3m, but is known on June 1.
# customer_id scores ~0.5: IDs carry no information here, but never use them as a feature (a model can memorise them).
print(leakage_scan(RAW_COLUMNS))
print(date_check())
print(train.groupby("churn")["retention_offer_sent"].mean().round(3))
""")

nb.exercise("1B – Repair a leaky feature (write code)", """
`region_churn_rate` could be a legitimate feature: the churn rate in a customer's region **as learned from the training data**.
The version in the extract was computed on all customers, including the test set.

1. Recompute the churn rate per postal code on the **training data only**, and map it onto the training and test rows.
   Postal codes that do not appear in the training data get the overall training churn rate.
   Hint: clean the postal code first with `df["region_code"].str.extract(r"(\\d{4})")[0]`.
2. Compare the test AUC of `BASE + ["region_churn_rate"]` (the leaky column) with `BASE + ["region_rate_train"]` (your version).
3. Why is even your version still slightly optimistic on the **training** data? (Think about customers in a postal code with only 2 customers.)
""")
nb.code("""
# your code here
""")
nb.solution("Solution – Exercise 1B", """
postal_train = train["region_code"].str.extract(r"(\\d{4})")[0]
postal_test = test["region_code"].str.extract(r"(\\d{4})")[0]
rate = y_train.groupby(postal_train).mean()                   # learned on the training data only
overall = y_train.mean()

Xtr, Xte = X_train.copy(), X_test.copy()
Xtr["region_churn_rate"], Xte["region_churn_rate"] = train["region_churn_rate"], test["region_churn_rate"]
Xtr["region_rate_train"] = postal_train.map(rate).fillna(overall)
Xte["region_rate_train"] = postal_test.map(rate).fillna(overall)

print(pd.Series({"base": test_auc(BASE, Xtr, Xte),
                 "base + leaky region_churn_rate": test_auc(BASE + ["region_churn_rate"], Xtr, Xte),
                 "base + region_rate_train": test_auc(BASE + ["region_rate_train"], Xtr, Xte)}).round(3))

# 3. On the training rows, each customer's own label is still part of the rate of its postal code. For small postal
#    codes that is a large share, so the model overestimates the feature. Remedies: out-of-fold encoding and smoothing,
#    exactly what scikit-learn's TargetEncoder does (see Notebook 3).
""")

nb.md("""
## 1.7 Take-aways

- Define the **prediction point** first; every feature must be computable on that day.
- Split **before** anything is learned from the data, and remove duplicates first.
- Screening tools find suspects, not culprits: a leak is proven by asking *when* and *how* a value was recorded.
- Leakage also enters through preprocessing: an aggregate computed on all data (like a regional churn rate) leaks the target.

**Back to the slides: debrief of the leakage detective.**
""")

nb.save("../notebooks")
