# Frontend

The React + Vite frontend sends patient encounter data to the FastAPI backend, displays the returned prediction, and loads recent saved prediction history. The trained model runs only on the backend.

## Local setup

From this directory:

```bash
cp .env.example .env
npm ci
npm run dev -- --host 0.0.0.0
```

Vite serves the app at `http://localhost:5173`. Set `VITE_API_BASE_URL` in `.env` when the API is not at `http://localhost:8000`.

## Tests and build

```bash
npm test
npm run build
```

The production output is written to `dist/`.
