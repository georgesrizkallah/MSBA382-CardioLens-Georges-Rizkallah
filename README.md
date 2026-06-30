# CardioLens — Cardiovascular Risk Dashboard (MSBA382)

Streamlit dashboard analysing 10-year coronary heart disease (CHD) risk using the
Framingham Heart Study, with global/Lebanon context from WHO, GBD (IHME) and IDF.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Password: **heart2026**

## Publish (Streamlit Community Cloud — free, ~5 minutes)
1. Push this `dashboard/` folder to a **public GitHub repo** (must include `app.py`,
   `chd_model.joblib`, `requirements.txt`, and the `data/` and `figures/` folders).
2. Go to https://share.streamlit.io → **New app** → pick the repo/branch → main file `app.py` → **Deploy**.
3. Copy the generated public URL 

## Sections (7)
Global Context · Population Explorer · Risk-Factor Analysis · Risk Calculator
(live SHAP) · Model & Validation · Preventable Burden · About

## Files
- `app.py` — the dashboard (7 sections + password gate, live cohort filters)
- `chd_model.joblib` — fitted models: calibrated logistic regression (used by the
  calculator), random forest, and an XGBoost model + SHAP background (used for the live
  per-patient explanation and the validation benchmarks)
- `data/` — cleaned Framingham cohort, global/Lebanon context tables, model-comparison
  metrics
- `figures/` — validation figures rendered in the Model & Validation section
- `requirements.txt` — Python dependencies (streamlit, scikit-learn, xgboost, shap, plotly…)
