from nbtools import NB

nb = NB("NB2_exploratory_data_analysis.ipynb")
nb.header("Notebook 2 – Exploratory data analysis",
          "2",
          "Explore a housing dataset before any modelling: profile it, study the target, find structural "
          "missingness, impossible values and redundant variables, and turn each finding into a decision.",
          30)

nb.md("""
## The case

An (artificial) housing dataset of 3,000 sales. The target is `SalePrice` (a regression problem).
No model is built in this notebook: the output of EDA is a list of **findings and decisions**.
""")
nb.data_cell("skrub missingno")
nb.code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

pd.set_option("display.width", 200)
""")

nb.md("""
## 2.1 Load the data and look at it

The file uses `;` as separator (a European CSV). A first sanity check: does every column have the type you expect?
""")
nb.code("""
house = pd.read_csv(DATA + "housing.csv", sep=";")
print(house.shape)
print(house.dtypes)
house.head()
""")

nb.md("""
## 2.2 Profiling

`describe()` gives the classic summary. `skrub.TableReport` gives an interactive overview of every column
(distribution, missing values, most frequent values, associations). Use it to *start* your analysis, not to end it.
""")
nb.code("""
house.describe().round(1)
""")
nb.code("""
from skrub import TableReport
TableReport(house)
""")

nb.md("""
## 2.3 The target variable

Is `SalePrice` symmetric, skewed or heavy-tailed? Would you transform it?
""")
nb.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(house["SalePrice"], bins=50)
axes[0].set_title(f"SalePrice (skewness = {house['SalePrice'].skew():.2f})")
axes[1].hist(np.log1p(house["SalePrice"]), bins=50)
axes[1].set_title(f"log(1 + SalePrice) (skewness = {np.log1p(house['SalePrice']).skew():.2f})")
for ax in axes: ax.set_ylabel("Number of houses")
plt.tight_layout(); plt.show()
""")
nb.md("""
The raw target is moderately right-skewed (about 0.66). The log transform over-corrects slightly.
For a linear regression, a log target can stabilise the variance, but the coefficients then become multiplicative.
For tree-based models the transformation is usually unnecessary.
""")

nb.md("""
## 2.4 Missing values: how much, where, and why?
""")
nb.code("""
missing = house.isna().mean().sort_values(ascending=False) * 100
missing[missing > 0].round(2).plot.bar(figsize=(6, 3), title="Missing values (% of rows)")
plt.ylabel("%"); plt.show()
msno.matrix(house, figsize=(10, 4), fontsize=9); plt.show()
""")
nb.md("""
**Why** is `PoolQuality` missing? Compare with `HasPool`.
""")
nb.code("""
pd.crosstab(house["HasPool"], house["PoolQuality"].isna(), rownames=["HasPool"], colnames=["PoolQuality missing"])
""")
nb.md("""
Every house without a pool has no pool quality: the missingness is **structural**, not random.
Dropping the column because "94% is missing" would throw away information; encode it as a category `"No pool"` instead.

Is `GarageArea` missing at random? Look at the missing share per number of garage places.
""")
nb.code("""
house.assign(garage_area_missing=house["GarageArea"].isna()).groupby("GarageCars")["garage_area_missing"].mean().round(3)
""")
nb.md("""
The share is about 10% for every garage size: consistent with values missing (completely) at random.
Median imputation is acceptable, or better, impute conditionally on `GarageCars`, which is almost the same information.
""")

nb.md("""
## 2.5 Impossible values

An area cannot be negative. How many negative garage areas are there, and where do they occur?
""")
nb.code("""
print("Negative GarageArea:", (house["GarageArea"] < 0).sum())
house.groupby("GarageCars")["GarageArea"].describe().round(1)
""")
nb.md("""
Houses without a garage have garage areas scattered around 0, including negative values: a recording artefact.
A defensible rule: set `GarageArea = 0` when `GarageCars = 0`. That is a *cleaning decision* that goes into your pipeline (Notebook 3).
""")

nb.md("""
## 2.6 Relationships with the target
""")
nb.code("""
corr = house.select_dtypes("number").corr()
print(corr["SalePrice"].drop("SalePrice").sort_values(ascending=False).round(3))
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=False)
plt.title("Correlation matrix (numeric features)"); plt.tight_layout(); plt.show()
""")
nb.code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(house["LivingArea"], house["SalePrice"], s=6, alpha=0.3)
axes[0].set_xlabel("Living area (sq ft)"); axes[0].set_ylabel("Sale price")
sns.boxplot(data=house, x="OverallQuality", y="SalePrice", ax=axes[1], color="lightgrey")
order = house.groupby("Neighborhood")["SalePrice"].median().sort_values().index
sns.boxplot(data=house, x="Neighborhood", y="SalePrice", order=order, ax=axes[2], color="lightgrey")
plt.tight_layout(); plt.show()
""")
nb.md("""
Correlation only captures *linear* relationships between *numeric* variables. `Neighborhood` does not appear in the
correlation matrix at all, yet the boxplot shows whether it matters. Weak correlation does not imply irrelevance.
""")

nb.md("""
## 2.7 Redundancy
""")
nb.code("""
print("Correlation GarageCars - GarageArea:", round(house[["GarageCars", "GarageArea"]].corr().iloc[0, 1], 3))
plt.figure(figsize=(5, 3.5))
plt.scatter(house["GarageCars"], house["GarageArea"], s=6, alpha=0.3)
plt.xlabel("Garage places (cars)"); plt.ylabel("Garage area"); plt.show()
""")
nb.md("""
Both variables carry nearly the same information. For a linear model this causes unstable coefficients (multicollinearity):
keep one. For tree-based models it matters much less.
""")

nb.md("""
## 2.8 Visualisation quality

The plot below is technically correct. Is it trustworthy, actionable and elegant?
""")
nb.code("""
plt.figure(figsize=(7, 4))
plt.scatter(house["LivingArea"], house["SalePrice"])
plt.grid(True, which="both", linewidth=1.2)
plt.title("LivingArea vs SalePrice")
plt.show()
""")

nb.exercise("2A – Improve the plot (run & tweak)", """
Change the parameters marked `# <- change` and rerun until you are happy with the plot.

1. What does transparency (`ALPHA`) reveal that the original plot hides?
2. Which elements can be removed without losing information?
3. Is the y-axis starting point trustworthy? Would you show this plot to a business stakeholder? With which title?
""")
nb.code("""
ALPHA = 1.0          # <- change, e.g. 0.2
POINT_SIZE = 20      # <- change, e.g. 6
SHOW_GRID = True     # <- change
TITLE = "LivingArea vs SalePrice"   # <- change: write the message, not the variable names

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(house["LivingArea"], house["SalePrice"], s=POINT_SIZE, alpha=ALPHA)
ax.grid(SHOW_GRID)
ax.set_title(TITLE)
ax.set_xlabel("LivingArea"); ax.set_ylabel("SalePrice")
plt.show()
""")
nb.solution("Solution – Exercise 2A", """
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(house["LivingArea"], house["SalePrice"] / 1000, s=6, alpha=0.2, color="tab:blue")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Living area (sq ft)")
ax.set_ylabel("Sale price (thousand EUR)")
ax.set_title("Larger houses sell for more, but the spread grows with size")
plt.show()
# Transparency shows where the mass of the data is; the heavy grid and frame add nothing.
""")

nb.exercise("2B – Let an AI assistant do the EDA, then review it (optional)", """
In Colab, open the Gemini panel (or the *Data Science Agent*) and ask: *"Perform an exploratory data analysis of the
dataframe `house` with SalePrice as target."* Then review the generated analysis with this checklist:

- Did it notice that `PoolQuality` is structurally missing, or did it propose to drop it?
- Did it detect the negative garage areas?
- Did it look at `Neighborhood` and `HouseStyle`, or only at correlations?
- Did it impute or transform anything **before** a train/test split?

Alternatively, write code yourself: make a plot that shows whether the effect of `LivingArea` on `SalePrice`
differs between `HouseStyle` categories.

*Privacy note:* only send data to an AI service when you are allowed to. This dataset is artificial.
""")
nb.code("""
# your code here
""")
nb.solution("Solution – Exercise 2B (plot)", """
g = sns.lmplot(data=house, x="LivingArea", y="SalePrice", col="HouseStyle", col_wrap=4,
               scatter_kws={"s": 4, "alpha": 0.2}, line_kws={"color": "black"}, height=2.6)
g.set_axis_labels("Living area (sq ft)", "Sale price"); plt.show()
""")

nb.md("""
## 2.9 From findings to decisions

Complete the table in your own copy (double-click to edit). This list is the real output of an EDA.

| Finding | Decision (and for which model type) |
|---|---|
| SalePrice moderately right-skewed | |
| PoolQuality missing when HasPool = No | |
| GarageArea missing (10%), negative values when GarageCars = 0 | |
| GarageCars and GarageArea correlated 0.99 | |
| Neighborhood not visible in the correlation matrix | |

**Back to the slides: debrief of the EDA exercise, then part 3.**
""")

nb.save("../notebooks")
