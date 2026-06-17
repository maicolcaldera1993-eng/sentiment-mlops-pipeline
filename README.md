---
title: Sentiment Analysis API
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Sentiment Analysis MLOps Pipeline

Sistema MLOps end-to-end per il monitoraggio della reputazione online:
classifica testi dai social in positive/neutral/negative usando un modello
pre-addestrato, con pipeline CI/CD automatizzata, deploy su Hugging Face e
monitoraggio via Grafana.

**Modello:** [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
**Demo live:** [Hugging Face Space](https://maicol931-sentiment-mlops-api.hf.space/docs)

## Architettura

| Componente | Tecnologia | Ruolo |
|------------|-----------|-------|
| Modello | RoBERTa (transformers) | classificazione del sentiment |
| API | FastAPI + uvicorn | model serving via HTTP |
| Test | pytest | unit, integrazione, end-to-end |
| CI/CD | GitHub Actions | test + valutazione + deploy automatici |
| Container | Docker | packaging dell'app |
| Deploy | Hugging Face Spaces | app pubblica |
| Monitoraggio | Grafana + Infinity | dashboard del sentiment in tempo reale |

## Struttura

```
src/sentiment.py      # nucleo: preprocessing + classificazione
app.py                # API FastAPI (endpoint /predict, /stats, /)
evaluate.py           # valutazione accuratezza su tweet_eval
simulate_traffic.py   # genera traffico per la demo di monitoraggio
tests/                # unit, integration, e2e
Dockerfile            # immagine dell'app
.github/workflows/    # pipeline CI/CD
monitoring/           # docker-compose + Grafana (provisioning + dashboard)
```

## Uso

### Eseguire l'app in locale
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
API disponibile su `http://localhost:8000` — documentazione interattiva su `/docs`.

### Endpoint
- `GET /` — health check
- `POST /predict` — classifica una lista di testi (`{"texts": ["..."]}`)
- `GET /stats` — statistiche aggregate delle predizioni (JSON)

### Eseguire i test
```bash
python -m pytest -v
```

### Valutare il modello
```bash
python evaluate.py
```
Calcola l'accuratezza su un campione di tweet_eval; fallisce se sotto il 60%.

### Monitoraggio con Grafana
```bash
# 1. avvia l'app (terminale 1)
uvicorn app:app --host 0.0.0.0 --port 8000
# 2. avvia Grafana (terminale 2)
cd monitoring && docker compose up
# 3. genera traffico (terminale 3)
python simulate_traffic.py
```
Grafana su `http://localhost:3000` — la dashboard "Distribuzione sentiment"
si popola in tempo reale. Fonte dati e dashboard sono caricate automaticamente
via provisioning.

## Risultati

- **Accuratezza** sul test set di tweet_eval (campione di 200): ~0.75
- **Pipeline CI/CD**: ogni push esegue test e valutazione; il deploy su
  Hugging Face avviene solo se entrambi superano i controlli.

## CI/CD

La pipeline (`.github/workflows/ci.yml`) ha tre job:
- `test` — unit, integration, e2e con pytest
- `evaluate` — accuratezza del modello su tweet_eval (quality gate)
- `deploy` — push automatico su Hugging Face Space, solo se `test` ed
  `evaluate` passano

## Documentazione

Le scelte progettuali, le motivazioni e i risultati dettagliati sono
descritti nel notebook di consegna (Google Colab).
