"""
train_model.py
---------------
Trains the supervised learning model used for diabetes risk assessment.

Model: Logistic Regression (chosen deliberately over a black-box model such as
a random forest or gradient booster) because its coefficients give an exact,
honest decomposition of *why* a given prediction came out the way it did.
That decomposition is what powers the "explanation" half of this project.

Data: This demo trains on a synthetically generated cohort rather than a
downloaded file, so the project runs anywhere with no external dependency.
The synthesis below follows the well-documented, real-world clinical risk
relationships for type 2 diabetes (fasting glucose is by far the strongest
predictor, followed by BMI, age, family history, and blood pressure) and adds
realistic feature correlation and label noise, producing a dataset with the
same 8 features and general shape as the classic Pima Indians Diabetes
dataset. Swap `make_dataset()` for a `pandas.read_csv(...)` of a real
clinical dataset and nothing else in this file needs to change.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

# Human-readable metadata used both for training-time sanity checks and by
# the API when it builds plain-language explanations.
FEATURE_META = {
    "pregnancies":      {"label": "Pregnancies",              "unit": ""},
    "glucose":          {"label": "Fasting glucose",          "unit": "mg/dL"},
    "blood_pressure":   {"label": "Blood pressure",           "unit": "mm Hg"},
    "skin_thickness":   {"label": "Triceps skinfold thickness","unit": "mm"},
    "insulin":          {"label": "Serum insulin",            "unit": "mu U/mL"},
    "bmi":              {"label": "Body mass index",          "unit": "kg/m²"},
    "diabetes_pedigree":{"label": "Family history score",     "unit": ""},
    "age":               {"label": "Age",                     "unit": "years"},
}


def make_dataset(n_samples: int = 6000, seed: int = 42):
    rng = np.random.default_rng(seed)

    age = rng.gamma(shape=6.0, scale=6.0, size=n_samples) + 18
    age = np.clip(age, 18, 90)

    pregnancies = rng.poisson(lam=np.clip((age - 18) / 10, 0.2, 6), size=n_samples)
    pregnancies = np.clip(pregnancies, 0, 17)

    bmi = rng.normal(loc=27 + (age - 40) * 0.03, scale=6.0, size=n_samples)
    bmi = np.clip(bmi, 15, 60)

    glucose = rng.normal(loc=100 + (bmi - 27) * 1.3 + (age - 40) * 0.25, scale=18, size=n_samples)
    glucose = np.clip(glucose, 55, 220)

    blood_pressure = rng.normal(loc=68 + (bmi - 27) * 0.6 + (age - 40) * 0.15, scale=10, size=n_samples)
    blood_pressure = np.clip(blood_pressure, 40, 130)

    skin_thickness = rng.normal(loc=20 + (bmi - 27) * 0.9, scale=8, size=n_samples)
    skin_thickness = np.clip(skin_thickness, 5, 60)

    insulin = rng.normal(loc=80 + (bmi - 27) * 4 + (glucose - 100) * 1.2, scale=45, size=n_samples)
    insulin = np.clip(insulin, 10, 550)

    diabetes_pedigree = rng.gamma(shape=1.8, scale=0.28, size=n_samples)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.05, 2.5)

    # Ground-truth risk relationship: glucose dominates, then BMI, age,
    # pedigree (family history) and blood pressure contribute more mildly.
    # Coefficients are on standardized-ish scales chosen to mimic published
    # odds ratios for these risk factors.
    z = (
        -1.55
        + 0.045 * (glucose - 100)
        + 0.085 * (bmi - 27)
        + 0.032 * (age - 40)
        + 1.05 * (diabetes_pedigree - 0.5)
        + 0.011 * (blood_pressure - 70)
        + 0.045 * pregnancies
        + 0.006 * (insulin - 100)
        + 0.010 * (skin_thickness - 20)
    )
    noise = rng.normal(0, 0.6, size=n_samples)
    outcome = (1 / (1 + np.exp(-(z + noise))) > 0.5).astype(int)

    X = np.column_stack([
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age,
    ])
    return X, outcome


def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("=== Diabetes Risk Model — training report ===")
    print(f"Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds, target_names=["Lower risk", "Higher risk"]))

    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump({"features": FEATURES, "meta": FEATURE_META,
                   "metrics": {"accuracy": acc, "roc_auc": auc}}, f, indent=2)

    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
