"""
generate_dataset.py
-------------------
Generates a synthetic student dataset with realistic distributions.
Target labels (Low / Medium / High risk) are rule-based and deterministic.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1200  # number of rows

# ── Feature generation ────────────────────────────────────────────────────────
attendance       = np.clip(np.random.normal(72, 15, N), 30, 100)
prev_gpa         = np.clip(np.random.normal(6.5, 1.5, N), 0, 10)
internal_marks   = np.clip(np.random.normal(55, 18, N), 0, 100)
backlogs         = np.random.choice([0, 1, 2, 3, 4, 5, 6], N,
                                    p=[0.35, 0.25, 0.18, 0.10, 0.07, 0.03, 0.02])

# ── Rule-based labelling ──────────────────────────────────────────────────────
# HIGH risk  → any of: attendance<55 | GPA<4.5 | marks<35 | backlogs>=4
# LOW risk   → all of: attendance≥75 | GPA≥7.0 | marks≥60 | backlogs==0
# MEDIUM     → everything else

def assign_risk(att, gpa, marks, bl):
    high = (att < 55) or (gpa < 4.5) or (marks < 35) or (bl >= 4)
    low  = (att >= 75) and (gpa >= 7.0) and (marks >= 60) and (bl == 0)
    if high:
        return "High"
    elif low:
        return "Low"
    else:
        return "Medium"

risk = [assign_risk(a, g, m, b)
        for a, g, m, b in zip(attendance, prev_gpa, internal_marks, backlogs)]

df = pd.DataFrame({
    "attendance":      np.round(attendance, 2),
    "prev_gpa":        np.round(prev_gpa, 2),
    "internal_marks":  np.round(internal_marks, 2),
    "backlogs":        backlogs,
    "risk_level":      risk
})

out_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
df.to_csv(out_path, index=False)

print(f"Dataset saved → {out_path}")
print(df["risk_level"].value_counts())
