"""Generate the expanded churn dataset (same schema as ExampleChurnDataset.xlsx, plus
minutes_used_two_months_ago). Planted issues:
leaky features, structural missingness, inconsistent formats, invalid/valid outliers,
duplicates and a high-cardinality postal code."""
import numpy as np, pandas as pd
rng = np.random.default_rng(2026)
N = 5000
PRED = pd.Timestamp("2023-06-01"); EXPORT = pd.Timestamp("2023-07-01")

# postal codes: long tail of ~250 Belgian codes, Leuven over-represented
codes = np.unique(np.r_[[1000,1020,1030,1050,1070,1080,1200,1300,1348,1400,1500,1800,2000,2018,2060,2100,2140,2170,2300,2400,2440,2500,2600,2800,2900,3000,3001,3010,3012,3018,3020,3050,3060,3070,3080,3090,3200,3290,3300,3500,3600,3700,3800,3900,4000,4020,4100,4300,4500,4800,5000,5100,6000,6200,7000,7100,7500,8000,8200,8400,8500,8800,9000,9040,9100,9200,9300,9400,9800,9900],
                        rng.choice(np.arange(1000,9990,10),200,replace=False)])
w = 1/np.arange(1,len(codes)+1)**0.9; rng.shuffle(w)
w[codes==3000] = w.max()*1.2; w = w/w.sum()
region = rng.choice(codes, N, p=w)
region_effect = dict(zip(codes, rng.normal(0,0.35,len(codes))))

prepaid = rng.random(N) < 0.35
start = pd.to_datetime("2014-01-01") + pd.to_timedelta(rng.integers(0, (pd.Timestamp("2023-05-15")-pd.Timestamp("2014-01-01")).days, N), unit="D")
tenure_y = (PRED - start).days/365.25
usage = np.exp(rng.normal(5.4, 0.9, N)) * np.where(prepaid, 0.35, 1.0)
help_prior = rng.poisson(0.6 + 0.8*(rng.random(N) < 0.25), N)
unpaid = np.where(prepaid, 0, rng.poisson(0.35, N))

logit = (-1.9 + 0.8*prepaid - 0.22*np.minimum(tenure_y, 8) + 0.45*help_prior + 0.6*unpaid
         - 0.35*(np.log1p(usage) - 5) + np.array([region_effect[c] for c in region]) + rng.normal(0, 0.9, N))
churn = (rng.random(N) < 1/(1+np.exp(-logit))).astype(int)

apr = usage * rng.uniform(0.9, 1.1, N)
may = np.where(churn==1, apr*rng.uniform(0.45, 1.05, N), apr*rng.uniform(0.85, 1.15, N))
jun = np.where(churn==1, may*rng.uniform(0, 0.06, N), may*rng.uniform(0.9, 1.1, N))      # LEAK: June usage
help_3m = help_prior + np.where(churn==1, rng.poisson(0.7, N), rng.poisson(0.15, N))     # window Apr-Jun: includes June
# last call date as seen at export (LEAK): active customers keep calling in June
last_call = np.where(churn==1,
    PRED - pd.to_timedelta(rng.integers(1, 60, N), unit="D"),
    np.where(rng.random(N) < 0.85, PRED + pd.to_timedelta(rng.integers(0, 30, N), unit="D"),
             PRED - pd.to_timedelta(rng.integers(1, 30, N), unit="D")))
early_jun = (churn==1) & (rng.random(N) < 0.12)
last_call = pd.to_datetime(last_call); last_call = last_call.where(~early_jun, PRED + pd.to_timedelta(rng.integers(0,5,N),unit="D"))
low = may < 3; last_call = last_call.where(~low, last_call - pd.to_timedelta(rng.integers(30,150,N),unit="D"))

bill = np.where(prepaid, np.nan, np.round(28 + 0.075*may + rng.normal(0, 6, N), 2))
bill = np.where(~prepaid & (rng.random(N) < 0.02), np.nan, bill)                         # a few MCAR gaps

df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, N+1)],
    "contract_type": np.where(prepaid, "Prepaid", "Postpaid"),
    "contract_start_date": start.strftime("%Y-%m-%d"),
    "last_call_date": pd.to_datetime(last_call).strftime("%Y-%m-%d"),
    "minutes_used_two_months_ago": np.round(apr).astype(int),
    "minutes_used_last_month": np.round(may).astype(int),
    "minutes_used_current_month": np.round(jun).astype(int),
    "helpdesk_calls_3m": help_3m,
    "avg_monthly_bill": bill,
    "unpaid_invoices": unpaid,
    "region_code": region.astype(str),
    "churn": churn})

# --- planted data-quality issues ---
def pick(frac): return rng.choice(N, int(frac*N), replace=False)
i = pick(0.03); df.loc[i, "contract_type"] = df.loc[i, "contract_type"].map({"Postpaid": "postpaid", "Prepaid": "PREPAID"})
i = pick(0.01); df.loc[i, "contract_type"] = df.loc[i, "contract_type"].replace({"Postpaid": "Post-paid", "postpaid": "Post-paid"})
leuv = np.where(df.region_code == "3000")[0]
j = rng.choice(leuv, int(0.3*len(leuv)), replace=False); df.loc[j, "region_code"] = rng.choice(["LEU-3000", "3000 Leuven"], len(j))
i = pick(0.01); df.loc[i, "region_code"] = " " + df.loc[i, "region_code"]
i = pick(0.02); df.loc[i, "contract_start_date"] = pd.to_datetime(df.loc[i, "contract_start_date"]).dt.strftime("%d/%m/%Y")
i = pick(0.003); df.loc[i, "minutes_used_last_month"] = -rng.integers(1, 200, len(i))          # invalid
i = pick(0.001); df.loc[i, "unpaid_invoices"] = 500                                            # invalid
i = rng.choice(np.where(~prepaid)[0], 4, replace=False); df.loc[i, "avg_monthly_bill"] = 99999.0 # invalid
i = pick(0.002); df.loc[i, "minutes_used_last_month"] = rng.integers(2500, 4000, len(i))        # valid heavy users
i = pick(0.002); df.loc[i, "helpdesk_calls_3m"] = rng.integers(15, 30, len(i))                  # valid extreme
dup = df.sample(40, random_state=1)
df = pd.concat([df, dup]).sample(frac=1, random_state=2).reset_index(drop=True)
df.to_csv("data/churn.csv", index=False)
print(df.shape, df.churn.mean().round(3), df.region_code.nunique())
