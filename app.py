"""
API di model serving per la sentiment analysis (FastAPI).

Espone il modulo src/sentiment.py come servizio HTTP:
- GET  /         -> health check (stato del servizio, per il monitoraggio)
- POST /predict  -> riceve una lista di testi, restituisce label e score

Avvio locale:  uvicorn app:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.sentiment import MODEL_NAME, load_model, analyze

import json
import logging
from collections import Counter as CollectionsCounter
from datetime import datetime, timezone

class PredictionRequest(BaseModel):
    """Corpo della richiesta: una lista di testi (type str) da classificare."""
    texts: list[str]


class Prediction(BaseModel):
    """Singolo risultato: etichetta di sentiment (type str) e confidenza [0, 1] (type float)."""
    label: str
    score: float


class PredictionResponse(BaseModel):
    """Corpo della risposta: un risultato per ogni testo ricevuto."""
    predictions: list[Prediction]


# Applicazione e caricamento del modello.

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifica testi social in positive/neutral/negative "
                f"usando {MODEL_NAME}",
    version="1.0.0",
)

classifier = load_model()   # eseguito all'import del modulo = all'avvio del server

# Stato di monitoraggio (in memoria finche' il processo e' acceso).

# Conteggio cumulativo delle predizioni per ciascuna label.
label_counts = CollectionsCounter()
# Lista degli score, per calcolare la confidenza media.
confidence_scores = []

# Logging strutturato: ogni predizione viene scritta come riga JSON su file.

logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(message)s",   # scrive il  JSON, senza decorazioni
)
prediction_logger = logging.getLogger("predictions")

@app.get("/")
def health_check():
    """Health check: conferma che il servizio e' vivo e dichiara il modello
    in uso. Sara' interrogato dal sistema di monitoraggio."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classifica i testi, aggiorna le statistiche e logga ogni predizione."""
    results = analyze(request.texts, classifier)
    for text, r in zip(request.texts, results):
        label = r["label"].lower()
        score = r["score"]
        # aggiorna lo stato in memoria
        label_counts[label] += 1
        confidence_scores.append(score)
        # scrive una riga di log JSON con timestamp
        prediction_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "label": label,
            "score": round(score, 4),
        }))
    return {"predictions": results}

@app.get("/stats")
def stats():
    """Statistiche aggregate delle predizioni, in JSON.
    E' la fonte dati che Grafana (plugin Infinity) interroghera'."""
    total = sum(label_counts.values())
    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0.0
    )
    return {
        "total_predictions": total,
        "positive": label_counts.get("positive", 0),
        "neutral": label_counts.get("neutral", 0),
        "negative": label_counts.get("negative", 0),
        "avg_confidence": round(avg_confidence, 4),
    }