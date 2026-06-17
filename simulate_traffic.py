"""
Generatore di traffico per la demo di monitoraggio.

Pesca un campione casuale di tweet dal dataset tweet_eval e li invia
all'endpoint /predict dell'app, simulando un flusso di feedback reali
dai social. Serve a popolare la dashboard Grafana con dati realistici.

Uso (con l'app in esecuzione su localhost:8000):
    python simulate_traffic.py
"""

import time
import random
import requests
from datasets import load_dataset

API_URL = "http://localhost:8000/predict"
SAMPLE_SIZE = 200
DELAY_SECONDS = 0.3   # pausa tra una richiesta e l'altra, per vedere
                      # la dashboard aggiornarsi gradualmente

def main():
    print("Caricamento tweet da tweet_eval...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")

    # Mescola e prende SAMPLE_SIZE tweet a caso.
    indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
    texts = [dataset[i]["text"] for i in indices]

    print(f"Invio di {SAMPLE_SIZE} tweet all'app (uno ogni {DELAY_SECONDS}s)...")
    print("Apri Grafana e guarda i pannelli aggiornarsi.\n")

    for n, text in enumerate(texts, start=1):
        try:
            response = requests.post(API_URL, json={"texts": [text]}, timeout=10)
            if response.status_code == 200:
                label = response.json()["predictions"][0]["label"]
                print(f"[{n}/{SAMPLE_SIZE}] {label:>8} <- {text[:50]}")
            else:
                print(f"[{n}] errore HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"[{n}] richiesta fallita: {e}")
        time.sleep(DELAY_SECONDS)

    print("\nSimulazione completata. La dashboard mostra l'aggregato.")

if __name__ == "__main__":
    main()