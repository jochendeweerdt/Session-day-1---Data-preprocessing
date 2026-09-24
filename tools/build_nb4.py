from nbtools import NB

nb = NB("NB4_anomaly_detection.ipynb")
nb.header("Notebook 4 – Anomaly detection",
          "4",
          "Compare univariate rules, the Mahalanobis distance, kNN distance, the Local Outlier Factor and Isolation Forest "
          "on credit card transactions, and evaluate them the way an investigation team would: how many of the top-k alerts are fraud?",
          20)

nb.md("""
In cleaning (part 3) an outlier was a problem to fix. Here the outlier **is** the signal.
Anomaly detection is unsupervised: it describes normal behaviour and flags what deviates from it. We keep the fraud
labels aside and only use them at the end, to play the role of the investigators who check the alerts.
""")
nb.data_cell("pyod")
nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
rng = np.random.default_rng(0)
""")

nb.md("""
## 4.1 Global and local anomalies on a toy example

Two clusters: a spread-out one and a dense one. Point **G** is far from everything (global anomaly).
Point **L** is close to the dense cluster in absolute terms, but far compared to how tightly that cluster is packed (local anomaly).
""")
nb.code("""
sparse = rng.normal([25, 45], 10, size=(60, 2))
dense = rng.normal([70, 20], 1.2, size=(40, 2))
toy = pd.DataFrame(np.vstack([sparse, dense, [[8, 95], [64, 20]]]), columns=["x1", "x2"])
labels = {len(toy) - 2: "G", len(toy) - 1: "L"}

K = 5          # <- change (number of neighbours)
knn_dist = NearestNeighbors(n_neighbors=K + 1).fit(toy).kneighbors(toy)[0][:, 1:].mean(axis=1)
lof = -LocalOutlierFactor(n_neighbors=K).fit(toy).negative_outlier_factor_

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, score, name in [(axes[0], knn_dist, f"kNN distance (k={K})"), (axes[1], lof, f"LOF (k={K})")]:
    sc = ax.scatter(toy["x1"], toy["x2"], c=score, cmap="Greys", edgecolor="black", s=40)
    for i, t in labels.items():
        ax.annotate(f"{t}: {score[i]:.1f}", toy.loc[i], xytext=(6, 6), textcoords="offset points")
    ax.set_title(name); plt.colorbar(sc, ax=ax)
plt.tight_layout(); plt.show()
print("Rank of L (1 = most anomalous): kNN", int((knn_dist > knn_dist[len(toy)-1]).sum()) + 1,
      "| LOF", int((lof > lof[len(toy)-1]).sum()) + 1)
""")
nb.md("""
kNN distance measures **absolute** isolation, so L looks normal next to the points of the sparse cluster.
LOF compares the density around a point with the density around its neighbours, so L stands out.
LOF values around 1 mean "as dense as my neighbours"; values clearly above 1 mean "less dense": possible anomaly.
""")

nb.md("""
## 4.2 Credit card transactions

A subsample of the well-known dataset of European card transactions (Dal Pozzolo et al., 2015): all 492 frauds and
30,000 legitimate transactions. `V1`–`V28` are **principal components** of the original (confidential) features,
`Time` is seconds since the first transaction and `Amount` is in EUR.
""")
nb.code("""
cc = pd.read_csv(DATA + "creditcard_sample.csv")
print(cc.shape, "| fraud rate:", round(cc["Class"].mean(), 4))
y = cc.pop("Class")            # labels are kept aside, used only for evaluation
cc.describe().T[["mean", "std", "min", "max"]].round(2).head(8)
""")
nb.md("""
Some preprocessing, reusing part 3: `Amount` is heavily skewed, so we take a log and scale it robustly.
`Time` itself is meaningless, but the hour of the day is not; as it is cyclical (23h is close to 0h), we encode it with sine and cosine.
""")
nb.code("""
X = cc.drop(columns=["Time", "Amount"]).copy()
X["log_amount"] = RobustScaler().fit_transform(np.log1p(cc[["Amount"]])).ravel()
hour = (cc["Time"] / 3600) % 24
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
print(X.shape)
""")

nb.md("""
## 4.3 Scores from five detectors

Every detector returns a score: higher = more anomalous. Scores are **not** probabilities.
""")
nb.code("""
scores = pd.DataFrame(index=X.index)

# univariate: absolute z-score of the amount
z = (cc["Amount"] - cc["Amount"].mean()) / cc["Amount"].std()
scores["z-score amount"] = z.abs()

# multivariate: Mahalanobis distance, classical and robust covariance
scores["Mahalanobis"] = EmpiricalCovariance().fit(X).mahalanobis(X)
scores["Mahalanobis (robust)"] = MinCovDet(random_state=0, support_fraction=0.9).fit(X).mahalanobis(X)

# distance and density based
K = 20                                                        # <- change
scores["kNN distance"] = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)[0][:, 1:].mean(axis=1)
scores["LOF"] = -LocalOutlierFactor(n_neighbors=K).fit(X).negative_outlier_factor_

# isolation based
scores["Isolation Forest"] = -IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
scores.describe().T[["mean", "50%", "max"]].round(2)
""")

nb.md("""
## 4.4 Evaluation with an investigation budget

Assume the fraud team can check **k** alerts. *Precision@k* = share of the top-k alerts that is fraud.
(The dataset contains 492 frauds; a random pick of k transactions would contain about 1.6% fraud.)
""")
nb.code("""
def precision_at_k(score, y, k):
    top = score.sort_values(ascending=False).index[:k]
    return y.loc[top].mean()

def evaluate(scores, y, ks=(50, 100, 250, 500)):
    rows = {name: {f"precision@{k}": precision_at_k(s, y, k) for k in ks} | {"AUC": roc_auc_score(y, s)}
            for name, s in scores.items()}
    return pd.DataFrame(rows).T.round(3)

evaluate(scores, y)
""")
nb.code("""
# How much do the detectors agree? Overlap of their top-500 alerts
top = {n: set(s.sort_values(ascending=False).index[:500]) for n, s in scores.items()}
pd.DataFrame({a: {b: len(top[a] & top[b]) for b in top} for a in top})
""")
nb.md("""
Observations to discuss:
- A single variable (the amount) catches almost nothing: fraud is not simply "large amounts".
- Multivariate methods do much better; the robust Mahalanobis distance is not fooled by the outliers it tries to find.
- LOF does poorly here. Frauds often come in small groups of similar transactions, so each fraud has dense
  fraudulent neighbours and does not look *locally* unusual (masking). A larger neighbourhood helps (Exercise 4A).
- Methods flag partly different transactions. Anomaly ≠ fraud: many alerts are just unusual, legitimate behaviour.
""")

nb.exercise("4A – Tune the detectors to your budget (run & tweak)", """
1. Change `K_INVESTIGATE` (e.g. 50 vs 1000). Does the ranking of the methods change?
2. Change `N_NEIGHBORS` for LOF, and `MAX_SAMPLES` for Isolation Forest. How sensitive are the results?
3. Which detector would you put in production, and what would you tell the fraud team about its alerts?
""")
nb.code("""
K_INVESTIGATE = 250      # <- change
N_NEIGHBORS = 20         # <- change
MAX_SAMPLES = 256        # <- change (Isolation Forest sub-sample size)

tuned = pd.DataFrame({
    "LOF": -LocalOutlierFactor(n_neighbors=N_NEIGHBORS).fit(X).negative_outlier_factor_,
    "Isolation Forest": -IsolationForest(n_estimators=200, max_samples=MAX_SAMPLES, random_state=0).fit(X).score_samples(X),
    "Mahalanobis (robust)": scores["Mahalanobis (robust)"],
}, index=X.index)
evaluate(tuned, y, ks=(K_INVESTIGATE,))
""")
nb.solution("Solution – Exercise 4A", """
for k in (50, 250, 1000):
    print(evaluate(scores, y, ks=(k,)).iloc[:, 0].sort_values(ascending=False).round(3).to_dict())
# Rankings can change with the budget: some methods are very precise at the very top, others degrade slowly.
# LOF is very sensitive to the number of neighbours: precision@250 is ~0.02 for k=5, ~0.15 for k=100 and ~0.36 for k=300,
# because frauds form small dense groups (masking). Isolation Forest is fairly stable.
# Message to the fraud team: alerts are unusual transactions, not proven fraud; confirmed cases should be
# fed back as labels, which later allows a supervised model (see the session on fraud analytics).
""")

nb.exercise("4B – More detectors and an ensemble (optional, code)", """
The PyOD library offers 60+ detectors with the same interface (`fit`, then `decision_scores_`).
Try `ECOD` (parameter-free, based on empirical tail probabilities) and one other detector of your choice.
Then build a simple ensemble: average the **ranks** of the scores of several detectors. Does it beat the best single detector?
""")
nb.code("""
from pyod.models.ecod import ECOD
# your code here
""")
nb.solution("Solution – Exercise 4B", """
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

ecod = ECOD().fit(X)
copod = COPOD().fit(X)
more = scores.copy()
more["ECOD (PyOD)"] = ecod.decision_scores_
more["COPOD (PyOD)"] = copod.decision_scores_
members = ["Mahalanobis (robust)", "Isolation Forest", "ECOD (PyOD)"]
more["Ensemble (mean rank)"] = more[members].rank().mean(axis=1)
evaluate(more, y)
""")

nb.md("""
## 4.5 Take-aways

- Scale and transform features first: distance- and density-based detectors depend on it.
- kNN distance finds global anomalies, LOF also finds local ones, Isolation Forest is fast and robust.
- Evaluate against the investigation capacity (precision@k), and treat alerts as leads, not as proof.

**Back to the slides: part 5, Conclusion.**
""")

nb.save("../notebooks")
