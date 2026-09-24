"""Subsample of the credit card fraud dataset (Dal Pozzolo et al., 2015; ULB Machine Learning Group,
Kaggle 'creditcardfraud', Open Database License). Keeps all 492 frauds and 30,000 random
legitimate transactions (fraud rate ~1.6% instead of 0.17%) so LOF/kNN run quickly in Colab."""
import pandas as pd
d = pd.read_csv("../raw/creditcard.csv")
s = pd.concat([d[d.Class == 1], d[d.Class == 0].sample(30000, random_state=42)]).sort_values("Time")
s = s.round({c: 4 for c in s.columns if c.startswith("V")})
s.to_csv("data/creditcard_sample.csv", index=False)
print(s.shape, s.Class.mean().round(4))
