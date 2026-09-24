# Diabetes Risk Check

A small full-stack project for early risk awareness: a person enters a handful of
common health measurements, and gets back both an estimated diabetes risk score
**and** a plain-language explanation of exactly which values pushed that score up
or down — so the tool supports an informed decision, not just a number.

```
diabetes-risk-app/
├── backend/
│   ├── app.py              # Flask API (GET /api/health, POST /api/predict)
│   ├── train_model.py      # Trains the supervised learning model
│   ├── requirements.txt
│   └── model/               # Created on first run: model.joblib, scaler.joblib, feature_meta.json
└── frontend/
    └── index.html           # Self-contained UI (HTML + CSS + JS, no build step)
```

## How it works

**Model.** A logistic regression classifier trained with scikit-learn on the
same 8 clinical features used in the well-known Pima Indians Diabetes dataset
(pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, family
history score, age). Logistic regression was chosen deliberately over a more
"powerful" black-box model: its coefficients give an *exact* mathematical
breakdown of how much each input contributed to a specific prediction, which
is what makes honest, faithful explanations possible instead of approximated
after-the-fact guesses.

The demo trains on a synthetically generated cohort (see the docstring in
`train_model.py`) so the project runs immediately with no dataset download.
Swap in a real clinical CSV via `pandas.read_csv(...)` in `make_dataset()` if
you have one — nothing else in the pipeline needs to change.

**Explanation.** For a logistic model, `logit(p) = intercept + Σ(coefficient_i × scaled_value_i)`.
The API computes each `coefficient_i × scaled_value_i` term directly from the
trained model and ranks the results, so the top few factors shown to the user
are the model's own reasoning, in order of magnitude — not a separate
heuristic layered on top.

**API.**
- `GET /api/health` — service + model status
- `POST /api/predict` — body: `{ pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age }`
  returns risk score, risk category (low/moderate/high), a ranked list of
  per-feature contributions, a one-line plain-language summary, and a
  disclaimer.

**Frontend.** A single static HTML file: an intake form on the left, and a
results panel on the right with an arc gauge, risk category, plain-language
summary, and a ranked bar-chart-style breakdown of contributing factors.

## Running it

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
python3 app.py
# First run trains the model automatically (a few seconds) and prints
# accuracy/ROC-AUC, then serves the API on http://127.0.0.1:5000

# 2. Frontend
# Just open frontend/index.html in a browser (or serve it with any static
# file server). It calls http://127.0.0.1:5000 by default — change the
# API_BASE constant near the bottom of index.html if you run the backend
# elsewhere.
```

## Notes on the model's honesty

- The training report printed on first run (`accuracy`, `roc_auc`) is real,
  computed on a held-out test split — not decorative.
- The API validates input ranges server-side and returns a 400 with a clear
  message on bad input.
- Every response includes a disclaimer that this is an educational estimate,
  not a diagnosis — shown to the user, not just documented in code.
- To move this from demo to production you'd want: a real clinical dataset,
  a proper WSGI server (gunicorn/uwsgi) instead of Flask's dev server, input
  auth/rate-limiting, and clinical review of the risk thresholds.
