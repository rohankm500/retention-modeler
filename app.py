import numpy as np
import pandas as pd
import joblib
import streamlit as st

FEATURES = ["age", "tenure", "past_claims", "annual_premium", "premium_increase"]
st.set_page_config(page_title="Premium Increase Simulator", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/policies.csv")

model, df = load_model(), load_data()

def simulate(increase):
    X = df[FEATURES].copy()
    X["premium_increase"] = increase
    retained = 1 - model.predict_proba(X)[:, 1]
    revenue = (retained * df["annual_premium"] * (1 + increase)).sum()
    claims = (retained * df["expected_claims"]).sum()   # claims don't rise with price
    return {"retention": retained.mean(), "revenue": revenue,
            "lost": (1 - retained).sum(), "claims": claims,
            "profit": revenue - claims, "loss_ratio": claims / revenue}

st.title("Premium Increase Simulator")
st.write("Move the slider to see projected retention and revenue for a rate change.")

inc = st.slider("Premium increase (%)", 0.0, 15.0, 5.0, 0.5) / 100
base, scen = simulate(0.0), simulate(inc)

c1, c2, c3 = st.columns(3)
c1.metric("Projected retention", f"{scen['retention']:.1%}",
          f"{(scen['retention'] - base['retention']) * 100:+.1f} pts vs. no increase")
c2.metric("Projected revenue", f"${scen['revenue']:,.0f}",
          f"{scen['revenue'] - base['revenue']:+,.0f} vs. no increase")
c3.metric("Expected policies lost", f"{scen['lost']:,.0f}")
c4, c5, c6 = st.columns(3)
c4.metric("Projected underwriting profit", f"${scen['profit']:,.0f}",
          f"{scen['profit'] - base['profit']:+,.0f} vs. no increase")
c5.metric("Loss ratio", f"{scen['loss_ratio']:.1%}",
          f"{(scen['loss_ratio'] - base['loss_ratio']) * 100:+.1f} pts vs. no increase",
          delta_color="inverse")
c6.metric("Expected claims cost", f"${scen['claims']:,.0f}")

# Revenue curve across all increases
grid = np.arange(0, 0.3001, 0.005)
curve = pd.DataFrame([{"Increase (%)": x * 100, **simulate(x)} for x in grid])
best = curve.loc[curve["revenue"].idxmax()]
best_p = curve.loc[curve["profit"].idxmax()]

left, right = st.columns(2)
with left:
    st.subheader("Revenue vs. premium increase")
    st.line_chart(curve.set_index("Increase (%)")["revenue"])
    st.info(f"Revenue-maximizing increase: **{best['Increase (%)']:.1f}%** "
            f"(${best['revenue']:,.0f}, retention {best['retention']:.1%})")
with right:
    st.subheader("Underwriting profit vs. premium increase")
    st.line_chart(curve.set_index("Increase (%)")["profit"])
    st.success(f"Profit-maximizing increase: **{best_p['Increase (%)']:.1f}%** "
               f"(${best_p['profit']:,.0f}, loss ratio {best_p['loss_ratio']:.1%})")
# Who leaves? Segment breakdown
st.subheader("Churn risk by age group at this increase")
X = df[FEATURES].copy()
X["premium_increase"] = inc
seg = df.assign(p_churn=model.predict_proba(X)[:, 1],
                age_band=pd.cut(df["age"], [17, 30, 45, 60, 80]).astype(str))
st.bar_chart(seg.groupby("age_band")["p_churn"].mean())