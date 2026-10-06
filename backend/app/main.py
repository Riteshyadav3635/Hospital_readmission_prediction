from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.prediction import router as prediction_router
from app.core.config import CORS_ORIGINS
from app.api.routes.prediction import health_check
from app.ml.predictor import get_model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_bundle()
    yield


app = FastAPI(
    title="Hospital Readmission Prediction API",
    version="1.0.0",
    description="Predict 30-day readmission risk using the existing trained logistic regression and PCA pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)

app.add_api_route("/health", health_check, methods=["GET"], tags=["health"])


@app.get("/")
def root():
    return {"message": "Hospital Readmission Prediction API", "status": "online"}
