import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 10_000

age = rng.integers(18, 80, n)
tenure = rng.integers(0, 20, n)                 # years with the company
past_claims = rng.poisson(0.4, n)
annual_premium = (rng.normal(1200, 300, n).clip(400) + past_claims * 150).round(2)
premium_increase = rng.uniform(0, 0.15, n)       # historical increases, 0–15%
risk_mult = 1 + 0.25 * (past_claims - 0.4)
expected_claims = (0.62 * annual_premium * risk_mult * rng.lognormal(0, 0.2, n)).round(2)

# The "true" churn behavior you're baking in
logit = (
    -2.2
    + 6.0 * premium_increase          # bigger increase -> more churn
    - 0.02 * (age - 45)               # older customers stickier
    - 0.08 * (tenure - 10)            # loyal customers stickier
    + 0.3 * (past_claims - 0.4)       # claimants shop around
)
p_churn = 1 / (1 + np.exp(-logit))
churned = rng.binomial(1, p_churn)

df = pd.DataFrame({
    "age": age, "tenure": tenure, "past_claims": past_claims,
    "annual_premium": annual_premium, "premium_increase": premium_increase.round(4),
    "expected_claims": expected_claims,
    "churned": churned,
})
df.to_csv("data/policies.csv", index=False)
print(df.describe())
print("Churn rate:", df.churned.mean())
print("Overall loss ratio:", round(df.expected_claims.sum() / df.annual_premium.sum(), 3))