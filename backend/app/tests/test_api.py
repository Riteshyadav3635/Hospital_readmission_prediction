from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes.prediction import get_db
from app.database.base import Base
from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        session = test_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_health_endpoints_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 200


def test_real_prediction_is_saved_and_available_in_history(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"patient_reference": "api-test-001"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["patient_reference"] == "api-test-001"
    assert result["prediction"] in (0, 1)
    assert 0 <= result["probability"] <= 1
    assert len(result["model_features"]) == 41

    history_response = client.get("/api/predictions?limit=10")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 1
    assert history[0]["patient_reference"] == "api-test-001"
    assert history[0]["prediction"] == result["prediction"]
