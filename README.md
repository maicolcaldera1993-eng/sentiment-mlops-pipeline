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

End-to-end MLOps project for monitoring a company's online reputation
through automated sentiment analysis of social media posts.

The model classifies texts as positive, neutral or negative using
[cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
served as a FastAPI application, tested and deployed automatically via a
CI/CD pipeline (GitHub Actions) and containerized with Docker.

## API endpoints
- `GET /` — health check (service status and model in use)
- `POST /predict` — classify a list of texts; returns label and score for each
- `GET /docs` — interactive API documentation (Swagger UI)

## Status
- [x] Phase 1 — Sentiment model implementation
- [x] Phase 2 — CI/CD pipeline (testing + automated deployment)
- [ ] Phase 3 — Deployment & monitoring

*Full project documentation (design choices, architecture, limitations) to be completed.*