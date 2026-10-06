# Hospital Readmission Prediction

A full-stack demonstration application that sends patient encounter fields to a FastAPI service, runs the project's saved readmission model artifacts on the server, saves the result, and displays the result and recent prediction history in a React web interface.

This repository uses the existing trained model and preprocessing artifacts. It does not train a replacement model or claim model accuracy or clinical performance.

## Features

- React form with validation and an API-backed prediction result.
- Recent saved predictions loaded from the backend.
- FastAPI endpoints for health, prediction, and prediction history.
- PostgreSQL support and SQLite fallback for local development.
- Alembic-managed database schema.
- Original scaler, PCA, logistic-regression model, and label encoders loaded by the backend.
- Docker Compose configuration for the frontend, backend, and PostgreSQL.

## Technology stack

- Frontend: React 18, TypeScript, Vite, Tailwind CSS, React Hook Form, Zod, Axios.
- Backend: Python 3.13, FastAPI, Pydantic 2, SQLAlchemy, Alembic.
- ML: pandas, NumPy, scikit-learn, joblib, and the saved pickle artifacts under `ml/models/`.
- Database: SQLite (default) or PostgreSQL 16.
- Tests: pytest, Vitest, and Testing Library.

## Project structure

```text
.
├── backend/
│   ├── alembic/                 # Database migrations
│   ├── app/
│   │   ├── api/routes/          # Health, prediction, and history API
│   │   ├── core/                # Environment configuration
│   │   ├── database/            # SQLAlchemy models and repository
│   │   ├── ml/                  # Saved-artifact prediction pipeline
│   │   ├── schemas/             # Pydantic request and response schemas
│   │   ├── services/
│   │   └── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── data/                        # Original dataset
├── frontend/
│   ├── src/                     # React application and frontend test
│   ├── Dockerfile
│   └── package.json
├── ml/
│   ├── models/                  # Original scaler, PCA, classifier, encoders
│   ├── notebooks/
│   └── reports/
├── .env.example                 # Docker Compose environment template
└── docker-compose.yml
```

## Prerequisites

- Python 3.13.
- Node.js 20 or newer and npm.
- Git for cloning/versioning the project.
- Docker Desktop with the `docker compose` plugin only if using the Docker workflow.

For the local SQLite workflow, a separate database server is not required. PostgreSQL is available as a local service or through the supplied Compose configuration.

## Environment variables

The backend reads `backend/.env` (when present) and process environment variables:

| Variable | Purpose | Local default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy connection string | Empty means SQLite at the project root in `predictions.db` |
| `APP_ENV` | Environment label | `development` |
| `CORS_ORIGINS` | Comma-separated allowed browser origins | `http://localhost:5173` |
| `PORT` | Port used by the Docker backend start command | `8000`; Render supplies its own value |

The frontend reads `frontend/.env` at Vite start:

| Variable | Purpose | Local default |
| --- | --- | --- |
| `VITE_API_BASE_URL` | Backend base URL used by browser requests | `http://localhost:8000` |

Docker Compose reads the root `.env` file:

| Variable | Purpose |
| --- | --- |
| `POSTGRES_DB` | PostgreSQL database name |
| `POSTGRES_USER` | PostgreSQL user |
| `POSTGRES_PASSWORD` | Required PostgreSQL password; set a private local value |
| `CORS_ORIGINS` | Allowed frontend origin(s) |
| `VITE_API_BASE_URL` | Backend URL reachable by the browser |

Do not commit `.env` files or use real credentials in a template. The checked-in `.env.example` intentionally leaves `POSTGRES_PASSWORD` blank; set it in your local ignored `.env` before starting Compose. For the current Compose connection URL, use an alphanumeric password so it does not require URL escaping.

## Database setup

### SQLite (default for local development)

SQLite is selected when `DATABASE_URL` is unset or empty. There is no separate database server to start. From the project root:

```bash
cp backend/.env.example backend/.env
cd backend
../.venv/bin/alembic upgrade head
```

With the example `DATABASE_URL` blank, the database is created at `predictions.db` in the project root. The file is ignored by Git.

### PostgreSQL

To start just the Compose PostgreSQL service, first configure the ignored root environment file:

```bash
cp .env.example .env
```

Set a private `POSTGRES_PASSWORD` in `.env`, then run:

```bash
docker compose up -d db
```

For the standalone backend, set `DATABASE_URL` in `backend/.env` to the reachable PostgreSQL URL, for example:

```dotenv
DATABASE_URL=postgresql+psycopg2://hospital_readmission:YOUR_LOCAL_PASSWORD@localhost:5432/hospital_readmission
```

Then run the migration from the backend directory:

```bash
cd backend
../.venv/bin/alembic upgrade head
```

To inspect the applied migration:

```bash
../.venv/bin/alembic current
```

Do not run the standalone PostgreSQL commands when Docker is unavailable unless you already have a PostgreSQL server running locally. Docker Compose starts PostgreSQL; the migration command is run by the backend container at startup.

## Backend setup

From the project root, create and activate the Python 3.13 environment, then install the declared dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Apply the schema migration and start the API (run these in a terminal from the project root):

```bash
cd backend
../.venv/bin/alembic upgrade head
../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend loads and validates the saved model artifacts during application startup. If the model files are missing or incompatible, startup fails rather than substituting dummy predictions.

## Frontend setup

In a second terminal from the project root:

```bash
cp frontend/.env.example frontend/.env
cd frontend
npm ci
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173`. The Vite dev server defaults to port `5173`; the frontend defaults to the API at `http://localhost:8000`.

## Run the project locally

1. Follow [Backend setup](#backend-setup) and keep the backend terminal running.
2. Follow [Frontend setup](#frontend-setup) in another terminal.
3. Visit `http://localhost:5173`, enter encounter fields, and submit a prediction.
4. The result is stored in the configured database and should appear in the Recent predictions table.

For SQLite, migrations must be applied before starting the backend. When `DATABASE_URL` points at PostgreSQL, make sure that server is running and reachable before applying migrations.

## Run with Docker

Docker Desktop and Docker Compose are required. From the project root:

```bash
cp .env.example .env
```

Edit `.env` and set a private `POSTGRES_PASSWORD`. Then build and start all three services:

```bash
docker compose up --build
```

The Compose backend waits for PostgreSQL health, applies `alembic upgrade head`, and starts FastAPI. The frontend is available at `http://localhost:5173`; the API is at `http://localhost:8000`. Stop the services with `Ctrl+C`; stop and remove the containers with `docker compose down`. The named `postgres_data` volume persists database contents. `docker compose down -v` removes that data volume.

**Verification note:** Docker Compose itself must be run on a machine with Docker installed. The project configuration and YAML can be inspected independently, but a successful local Compose run should not be assumed unless you execute the command above.

## API documentation

When the backend is running:

- Interactive Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health status |
| `GET` | `/api/health` | Health status (API-prefixed alias) |
| `POST` | `/api/predict` | Run a prediction and save its record |
| `GET` | `/api/predictions?limit=10` | List the most recent saved predictions |

The prediction request fields have defaults in the Pydantic schema; for example, a minimal valid request can contain `{"patient_reference":"demo-001"}`. The history response includes the patient reference, prediction, probability, risk level, and creation time. The full input feature mapping is saved with the record for inference traceability but is not returned by the history list endpoint.

## ML model and prediction flow

1. FastAPI validates the request and the backend applies the existing categorical encoders and mappings/defaults.
2. The fields are assembled in the exact order exposed by `scaler.pkl`.
3. The saved `StandardScaler` transforms the model input.
4. The saved PCA artifact transforms the scaled row.
5. The saved Logistic Regression model calculates class probabilities; the existing decision rule labels the result as positive when the positive-class probability is at least `0.5`.
6. The prediction and associated feature values are saved to `prediction_records`; the response is rendered by the frontend.

The backend loads these original files from `ml/models/`:

- `scaler.pkl`
- `pca_95.pkl`
- `hospital_readmission_model.pkl`
- `label_encoders.pkl`

The frontend does not load or execute these artifacts. Do not remove or replace them with randomly generated or placeholder predictions.

## Testing

Run the backend tests from the project root:

```bash
cd backend
PYTHONPATH=. ../.venv/bin/python -m pytest app/tests -q
```

Run the frontend automated tests and production build:

```bash
cd frontend
npm test
npm run build
```

The backend tests cover the real artifact prediction path, API health/OpenAPI behavior, database persistence, and history retrieval. The frontend test covers form submission, result rendering, and history refresh.

## Deployment

### Frontend — Vercel

Import the repository in Vercel and configure the frontend project with `frontend` as the Root Directory, framework preset Vite, build command `npm run build`, and output directory `dist`. Set `VITE_API_BASE_URL` to the public Render backend URL (for example, `https://your-service.onrender.com`, with no `/api` suffix). Vite embeds this public URL in the client build; do not put secrets in `VITE_*` variables.

### Backend — Render

Create a Render Web Service using the repository's Dockerfile with the repository root as the Docker build context and `backend/Dockerfile` as the Dockerfile path. The Docker image includes the backend source and saved model artifacts. Set:

- `DATABASE_URL` to the private/internal URL for the hosted PostgreSQL-compatible database.
- `CORS_ORIGINS` to the exact deployed Vercel origin (comma-separate additional origins if needed).
- `APP_ENV` to `production`.

The container applies Alembic migrations before starting the API and uses Render's `PORT` when provided. Do not place credentials in the Dockerfile, frontend code, or repository.

### Database — hosted PostgreSQL-compatible service

Create a PostgreSQL database with your selected provider and use its connection URL as the backend's `DATABASE_URL`. Configure the Render backend to use the provider's private connection string where available. Keep credentials in the hosting provider's secret/environment-variable settings, not in source control.

## Limitations and disclaimer

- This is a software/portfolio demonstration, not a validated clinical decision-support system. It must not be used to make or replace clinical decisions.
- The repository does not assert accuracy, calibration, fairness, or clinical performance. Validate the model, input definitions, and intended use independently before any real-world use.
- The model was trained on historical data and may not generalize to other populations, institutions, or current care practices.
- Prediction requests can include sensitive encounter attributes, and the database persists the mapped input features plus optional `patient_reference`. Do not submit real patient/PHI data to an unsecured or non-compliant deployment. Authentication, authorization, retention controls, and a formal privacy/security review are not implemented here.
- Docker Compose execution was not verified in environments without Docker. Follow the Docker verification note above and run `docker compose up --build` on a Docker-enabled machine before relying on that deployment path.
