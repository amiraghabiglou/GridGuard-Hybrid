import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "consumers": [
            {
                "consumer_id": "SGCC_999",
                "consumption_values": [10.5, 10.2, 0.0, 0.0, 0.0, 8.5, 9.0]
            }
        ]
    }


@patch("src.api.main.process_theft_analysis.delay")
def test_async_detection_submission(mock_celery, sample_payload):
    """Test that the API correctly hands off tasks to the worker queue."""
    mock_celery.return_value = MagicMock(id="test-job-uuid")

    response = client.post("/detect", json=sample_payload)

    assert response.status_code == 202
    assert response.json()["job_id"] == "test-job-uuid"
    mock_celery.assert_called_once()


@patch("src.workers.tasks.process_theft_analysis.AsyncResult")
def test_results_polling(mock_async_result):
    """Test the polling mechanism for retrieving completed analysis."""
    mock_job = MagicMock()
    mock_job.ready.return_value = True
    mock_job.result = [{"consumer_id": "SGCC_999", "risk_tier": "HIGH"}]
    mock_async_result.return_value = mock_job

    response = client.get("/results/test-job-uuid")

    assert response.status_code == 200
    assert response.json()["status"] == "Completed"
    assert response.json()["result"][0]["consumer_id"] == "SGCC_999"