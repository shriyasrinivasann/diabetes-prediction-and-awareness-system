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

The project trains on the included `backend/diabetes.csv` dataset (Pima Indians Diabetes cohort with 768 clinical samples). If `diabetes.csv` is absent, it seamlessly falls back to synthetic dataset generation in `train_model.py`.

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
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Train model (optional: app.py runs this automatically if model is not trained)
python train_model.py

# 3. Start Application Server
python app.py
# Serves both the Flask API and the Frontend interface at http://127.0.0.1:5000
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
