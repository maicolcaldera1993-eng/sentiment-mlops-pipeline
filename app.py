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


@app.get("/")
def health_check():
    """Health check: conferma che il servizio e' vivo e dichiara il modello
    in uso. Sara' interrogato dal sistema di monitoraggio."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classifica i testi ricevuti. Il preprocessing (menzioni, URL) e'
    applicato internamente da analyze()"""
    results = analyze(request.texts, classifier)
    return {"predictions": results}