import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample structured dataset (can expand later)
data = {
    "keyword_hits": [0,1,2,3,0,2,1,3,2,0,3,2,1,0,3,2],
    "metadata_flag": [0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1],
    "tamper_flag": [0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1],
    "label": [0,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1]
}

df = pd.DataFrame(data)

X = df[["keyword_hits", "metadata_flag", "tamper_flag"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")

print("✅ ML Model Trained & Saved Successfully")