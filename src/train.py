import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURES = ["age", "tenure", "past_claims", "annual_premium", "premium_increase"]

df = pd.read_csv("data/policies.csv")
X, y = df[FEATURES], df["churned"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression()),
    "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=50, random_state=42),
}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print(f"{name}: AUC={roc_auc_score(y_te, p):.3f}  Brier={brier_score_loss(y_te, p):.4f}")

joblib.dump(models["logreg"], "models/churn_model.pkl")
print("Saved model to models/churn_model.pkl")