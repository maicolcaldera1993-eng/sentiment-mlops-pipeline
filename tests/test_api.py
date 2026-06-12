"""
Test end-to-end dell'API FastAPI.
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check_is_alive():
    """GET / deve rispondere 200 e dichiarare stato e modello in uso"""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_predict_returns_valid_structure():
    """POST /predict con input valido: 200, una predizione per testo,
    ciascuna con label e score"""
    response = client.post("/predict", json={"texts": ["Great!", "Awful."]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    for p in predictions:
        assert "label" in p and "score" in p


def test_predict_classifies_sentiment():
    """Test end-to-end: una frase positiva deve uscire
    'positive' attraversando l'INTERA catena HTTP -> modello."""
    response = client.post("/predict", json={"texts": ["I love this product!"]})
    label = response.json()["predictions"][0]["label"]
    assert label.lower() == "positive"


def test_missing_field_is_rejected():
    """Richiesta senza il campo obbligatorio 'texts': la validazione
    Pydantic deve respingerla con 422 PRIMA che tocchi il modello."""
    response = client.post("/predict", json={"sentences": ["hello"]})
    assert response.status_code == 422


def test_wrong_type_is_rejected():
    """'texts' come stringa invece che lista: la validazione
    Pydantic deve respingerla con 422 PRIMA che tocchi il modello."""
    response = client.post("/predict", json={"texts": "not a list"})
    assert response.status_code == 422