# Hospital Readmission Prediction

## Overview

A Streamlit research application for exploring the project's diabetic inpatient encounter dataset and making inference with its saved StandardScaler, PCA, and Logistic Regression artifacts. The app uses the existing trained model and preprocessing implementation; it does not train a model at runtime.

## Problem Statement

The project formulates a binary classification task: predict whether an encounter belongs to the dataset's `<30` day readmission class. The existing notebook maps `<30` to class 1 and maps `NO` and `>30` to class 0. One earlier markdown cell in the notebook states the reverse mapping; the executable transformation and saved artifacts are the basis used by this application.

## Features

- Dashboard with counts derived from the bundled dataset, saved model, and local prediction history.
- Prediction form generated from the saved scaler's exact 41 input features and fitted encoder categories.
- Inference through the saved categorical encoders/mappings, StandardScaler, 35-component PCA, and Logistic Regression classifier.
- Interactive dataset filters and charts for target outcome, age, gender, admission type, discharge disposition, specialty, and encounter utilization.
- Interactive Plotly dashboard charts and live dataset-derived KPI cards and key insights.
- Prediction history stored in a local CSV with prediction outputs only; submitted feature values and identifiers are not stored.
- Dataset Explorer with shape, data types, missingness, unique-value counts, numeric summaries, search, paging, and filtered CSV download.
- Model Insights page with saved estimator/PCA details, PCA variance charts, and evaluation results recorded in the original notebook.
- CSV downloads for predictions, filtered analytics data, dataset records, and prediction history.

## Dataset

The repository contains `data/diabetic_data.csv`, a diabetic inpatient encounter dataset with 101,766 rows and 50 source columns. Its target column is `readmitted`, with source values `NO`, `>30`, and `<30`. The binary model target is `<30` versus the other two categories.

The source has `encounter_id` and `patient_nbr` identifiers. They are not model features and are excluded from the Dataset Explorer table and its downloads. The app does not use these identifiers in prediction history.

## Machine Learning Pipeline

```text
Encounter input
→ Existing category mappings and fitted LabelEncoders
→ Saved StandardScaler
→ Saved PCA transform (35 components)
→ Saved Logistic Regression classifier
→ Class and probability estimate
```

The model expects 41 features in the saved scaler's fitted order. Prediction categories are sourced from the existing encoder artifacts and mappings. Model objects are cached for the app process; the dataset is cached after its first load.

## Application Features

The sidebar provides seven sections: Dashboard, Predict Readmission, Analytics, Prediction History, Model Insights, Dataset Explorer, and About Project. Analytics and Explorer filters use the bundled CSV. The prediction form passes the complete, validated feature row through the existing `src.predictor.predict` function.

The classifier exposes classes 0 and 1, and the app displays the class 1 probability when available. This is the estimator's probability output, not a calibrated clinical risk score. The UI does not introduce low/moderate/high thresholds.

## Model Details

The tracked artifacts in `ml/models/` are the fitted label encoders, scaler, PCA transform, and Logistic Regression classifier. The saved classifier uses `class_weight="balanced"`, `solver="lbfgs"`, `C=1.0`, `max_iter=1000`, and `random_state=42`. The saved PCA has 35 components and retains approximately 95.74% of variance in the fitted data.

The analysis notebook records these baseline held-out results: accuracy 66.86%, class 1 precision 17.18%, class 1 recall 51.56%, class 1 F1 25.77%, ROC-AUC 64.54%, and PR-AUC 0.199. The notebook also records a confusion matrix and cross-validation results. These are notebook outputs, not metrics recalculated by the Streamlit app; see `ml/notebooks/Hospital_Readmission_Analysis.ipynb` for their evaluation context. They are not clinical validation results.

## Technology Stack

- Python 3.13
- Streamlit
- Plotly
- pandas
- NumPy
- scikit-learn
- joblib

All runtime packages are pinned in `requirements.txt`. No separate API server, database, Node.js, or secrets configuration is required.

## Project Structure

```text
.
├── app.py                         # Streamlit entry point and application pages
├── requirements.txt               # Pinned runtime dependencies
├── .streamlit/config.toml          # Streamlit theme and server settings
├── README.md
├── data/diabetic_data.csv          # Bundled source encounter data
├── src/
│   ├── predictor.py                # Loads artifacts and runs inference
│   ├── preprocessing.py            # Feature mappings and validation
│   └── utils.py                    # Prediction history helpers
├── ml/
│   ├── models/                     # Saved encoders, scaler, PCA, classifier
│   ├── notebooks/                  # Original analysis and evaluation outputs
│   └── reports/                    # Existing analysis visualizations
└── tests/                          # Predictor and Streamlit app tests
```

## Installation

Python 3.13 matches the serialized artifacts and dependency versions used by this project.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running Locally

From the repository root:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Prediction History

The app stores only timestamp, predicted class label, probability, and the corresponding model output label in the ignored `prediction_history.csv` file. It does not store submitted feature values or identifiers. The history is local to the running app and may be shared among visitors to the same deployment. Streamlit Community Cloud storage is ephemeral, so history may be lost when an app restarts or redeploys. This file is not a production medical database.

## Streamlit Cloud Deployment

1. Push this repository to GitHub. Keep `app.py`, `requirements.txt`, `.streamlit/config.toml`, and all four files under `ml/models/` in the selected branch.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the GitHub repository and branch.
4. Set the app file path to `app.py`.
5. In **Advanced settings**, choose Python 3.13. This project pins pandas 2.2.3, which is not compatible with the Python 3.14.8 runtime shown in the current deployment logs.
6. Deploy. Community Cloud installs packages from the root `requirements.txt` and reads `.streamlit/config.toml` automatically.

If an existing app was created with Python 3.14.8, Community Cloud does not let you change its Python version after deployment. Delete that app and create it again with Python 3.13 selected in **Advanced settings**.

No secrets or external services are required. Keep the model artifacts committed so the app can load them at startup.

## Limitations

- The dataset reflects historical encounters and may not represent current practice, another hospital, or another population.
- The model's positive class is the dataset's `<30` category. Class 0 combines `NO` and `>30`.
- Evaluation values are recorded in the project notebook and are not recomputed for a new test split by this app.
- The model output can be wrong. It is not a diagnosis, care recommendation, or clinically validated score.
- Prediction history is a temporary local CSV and is not suitable for protected health information or production use.
- Pickle/joblib artifacts must be treated as trusted files. Do not load artifacts from untrusted sources.

## Future Improvements

- Revalidate performance on a current, representative, independently governed dataset.
- Add durable, access-controlled storage only if an explicitly approved production use case requires it.
- Document clinical review, calibration, fairness, and monitoring before any clinical application.

## Tests

Run the automated app and predictor checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The suite checks artifact loading, feature validation, a real saved-model prediction, app page rendering, and local history behavior.

## Medical Disclaimer

This application is intended for educational and research purposes only. It is not a medical diagnosis and should not replace professional medical advice.
