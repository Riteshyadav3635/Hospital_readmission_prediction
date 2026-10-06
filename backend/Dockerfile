FROM python:3.13-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY ml/models/ ./ml/models/

EXPOSE 8000

WORKDIR /app/backend

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
