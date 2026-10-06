# Hospital Readmission Prediction

## Overview

A single-process Streamlit application for exploring the existing hospital readmission machine-learning model. The app accepts encounter features, applies the saved encoders and fitted preprocessing artifacts, and displays the trained Logistic Regression model's prediction. It does not train or substitute another model.

This repository did not contain an existing Streamlit entry point when this conversion began. The previous checked-in app was a React/FastAPI/PostgreSQL implementation; that infrastructure has been removed in favor of the requested Streamlit-only design.

## Features

- Home page with project purpose, pipeline, dataset context, and disclaimer.
- Prediction form built from the fitted scaler's actual 41 feature names.
- Category choices sourced from saved label encoders and the model's existing mappings.
- Prediction from the original scaler, PCA, and Logistic Regression artifacts.
- Probability display when the saved classifier exposes the positive-class probability.
- Session-local prediction history with a clear-history action.
- No separate server, database, secrets, or patient-input retention required.

## Machine Learning Approach

The original saved inference path uses the fitted encoders/mappings followed by a `StandardScaler`, PCA, and Logistic Regression. The scaler's stored feature names define input order; the saved artifacts are checked for compatible feature dimensions when loaded.

The app shows no accuracy or other model performance metrics because this inference application does not evaluate the model.

## Dataset

The repository preserves the source diabetic inpatient dataset at `data/diabetic_data.csv` (101,766 encounter rows; target column `readmitted`), along with the original analysis notebooks and reports. The app uses saved model artifacts for inference and does not load the full dataset at runtime.

The input form uses encounter fields represented in the model's saved scaler, not arbitrary UI-only fields. It deliberately does not collect patient identifiers.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn
- Joblib (to load the existing pickle-compatible artifacts)

Matplotlib and Plotly are not required by the application.

## Project Structure

```text
.
├── app.py                         # Streamlit entry point
├── requirements.txt               # Runtime dependencies only
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                # Streamlit appearance/server settings
├── src/
│   ├── predictor.py               # Loads artifacts and runs inference
│   ├── preprocessing.py           # Model feature mappings and validation
│   └── utils.py                   # Session-history helpers
├── ml/
│   ├── models/                    # Existing scaler, PCA, classifier, encoders
│   ├── notebooks/                 # Original analysis notebooks
│   └── reports/                   # Existing analysis outputs
├── data/
│   └── diabetic_data.csv
└── tests/
    └── test_predictor.py
```

## Installation

Use Python 3.13 to match the local environment used with the serialized scikit-learn artifacts and pinned dependencies.

Create and activate a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the app dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). No environment variables, database service, or API server are needed.

## Streamlit Cloud Deployment

1. Push this repository to GitHub, ensuring `app.py`, `requirements.txt`, and all files in `ml/models/` are committed.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the GitHub repository and branch, and set the app file path to `app.py` at the repository root.
4. Open **Advanced settings** and choose **Python 3.13** to match the project's local runtime and the pinned package versions.
5. Deploy. Community Cloud installs the dependencies from the root `requirements.txt` and reads `.streamlit/config.toml` automatically.

No secrets configuration is needed. Keep the saved model artifacts in the repository so the app can load them at startup. Streamlit Community Cloud's local runtime is ephemeral; session prediction history is temporary and can disappear after a restart or when a user session ends.

## Model Pipeline

```text
Data preprocessing
→ Feature encoding and mappings
→ Feature scaling (saved StandardScaler)
→ PCA (saved PCA model)
→ Logistic Regression (saved model)
→ Prediction and probability when supported
```

The application loads the original files from `ml/models/`:

- `label_encoders.pkl`
- `scaler.pkl`
- `pca_95.pkl`
- `hospital_readmission_model.pkl`

It checks that the scaler's feature list matches the PCA input dimension and that the PCA output dimension matches the classifier input dimension. It does not retrain, replace, or randomly simulate the model. Python pickle/joblib artifacts must be treated as trusted files; only use the artifacts provided by this project.

## Limitations

- Prediction history lives only in the current Streamlit session's memory. It is not permanent cloud storage and is not shared between users.
- The history records only timestamp, predicted class, probability, and risk label; it does not save submitted feature values or identifiers.
- The application depends on the bundled model artifacts and the pinned library versions. Do not change scikit-learn independently of the serialized artifacts without validating compatibility.
- Historical training data may not represent current practice, another institution, or a different patient population.
- The output is a model estimate and can be wrong. No clinical validation or performance claim is made here.

## Tests

The test suite uses Python's standard `unittest` runner and Streamlit's built-in app test harness. Run it from the repository root after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

## Medical Disclaimer

This application is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.

## GitHub Repository

Final GitHub URL: `TODO: add the final repository URL`

## Live Demo

Streamlit Community Cloud URL: `TODO: add the deployed app URL`
