"""
train_model.py - Train Random Forest + XGBoost (or GradientBoosting fallback)
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found - using GradientBoostingClassifier instead.")

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.getenv("STUDENT_DATA_PATH", os.path.join(BASE, "..", "data", "student_data.csv"))
MODEL  = os.path.join(BASE, "model.pkl")
SCALER = os.path.join(BASE, "scaler.pkl")
LE     = os.path.join(BASE, "label_encoder.pkl")
RF_IMP = os.path.join(BASE, "rf_importances.pkl")

if not os.path.exists(DATA):
    print("Generating dataset...")
    exec(open(os.path.join(BASE, "..", "data", "generate_dataset.py")).read())

df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows.\n{df['risk_level'].value_counts()}\n")

FEATURES = ["attendance", "prev_gpa", "internal_marks", "backlogs"]
X     = df[FEATURES].values
y_raw = df["risk_level"].values

le = LabelEncoder()
y  = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average="weighted")
print("Random Forest  =>  Accuracy:", round(rf_acc,4), " F1:", round(rf_f1,4))
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# Advanced model
if HAS_XGB:
    adv = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    adv_name = "XGBoost"
else:
    adv = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    adv_name = "GradientBoosting"

adv.fit(X_train, y_train)
adv_pred = adv.predict(X_test)
adv_acc  = accuracy_score(y_test, adv_pred)
adv_f1   = f1_score(y_test, adv_pred, average="weighted")
print(f"{adv_name}  =>  Accuracy:", round(adv_acc,4), " F1:", round(adv_f1,4))
print(classification_report(y_test, adv_pred, target_names=le.classes_))

best_model, best_name = (adv, adv_name) if adv_f1 >= rf_f1 else (rf, "Random Forest")
print(f"\nBest model: {best_name}")

importances = rf.feature_importances_
print("\nFeature Importances:")
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"  {feat:20s}: {imp:.4f}")

with open(MODEL,  "wb") as f: pickle.dump(best_model, f)
with open(SCALER, "wb") as f: pickle.dump(scaler, f)
with open(LE,     "wb") as f: pickle.dump(le, f)
with open(RF_IMP, "wb") as f: pickle.dump(dict(zip(FEATURES, importances)), f)
print("\nAll artifacts saved successfully!")
