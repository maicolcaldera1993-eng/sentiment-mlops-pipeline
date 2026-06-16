"""
Valutazione del modello di sentiment analysis sul dataset pubblico "tweet_eval".

Misura l'accuratezza confrontando le predizioni del modello con le etichette
vere (ground truth) di un campione del dataset. Se l'accuratezza scende sotto
una soglia minima, lo script termina con errore (exit code 1): cosi' puo'
essere usato in CI come "quality gate" del modello.
"""

import sys
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

from src.sentiment import load_model, analyze

# Soglia minima di accuratezza accettabile. 
ACCURACY_THRESHOLD = 0.60

# Quantità di esempi da valutare. 
SAMPLE_SIZE = 200

# tweet_eval usa etichette numeriche: 0=negative, 1=neutral, 2=positive.
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    print("Caricamento dataset tweet_eval (sentiment)...")

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=f"test[:{SAMPLE_SIZE}]")

    texts = dataset["text"]
    true_labels = [ID_TO_LABEL[label_id] for label_id in dataset["label"]]

    print(f"Valutazione su {len(texts)} esempi...")
    classifier = load_model()
    results = analyze(texts, classifier)
    predicted_labels = [r["label"].lower() for r in results]

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuratezza: {accuracy:.4f}")
    print("\nReport per classe:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    if accuracy < ACCURACY_THRESHOLD:
        print(f"FALLITO: accuratezza {accuracy:.4f} sotto la soglia "
              f"{ACCURACY_THRESHOLD}")
        sys.exit(1)
    print(f"OK: accuratezza sopra la soglia {ACCURACY_THRESHOLD}")


if __name__ == "__main__":
    main()