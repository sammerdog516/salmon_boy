from __future__ import annotations


def test_risk_tiles_model_source_without_predictions_returns_400(client) -> None:
    response = client.get("/risk/tiles", params={"source": "model"})
    assert response.status_code == 400
    assert "No model predictions found" in response.text


def test_risk_predict_missing_scene_returns_400(client) -> None:
    response = client.post("/risk/predict", json={"scene_id": "scene-does-not-exist"})
    assert response.status_code == 400
    assert "Scene not found" in response.text

