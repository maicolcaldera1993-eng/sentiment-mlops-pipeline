from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def preprocess(text):
    """Normalizza il testo come da model card: @menzioni -> @user, URL -> http."""
    tokens = []
    for t in text.split(" "):
        if t.startswith("@") and len(t) > 1:
            t = "@user"
        elif t.startswith("http"):
            t = "http"
        tokens.append(t)
    return " ".join(tokens)

def load_model():
    """Carica la pipeline di sentiment analysis."""
    return pipeline("sentiment-analysis", model=MODEL_NAME)

def analyze(texts, classifier=None):
    """Classifica una lista di testi. Ritorna lista di dict con label e score."""
    if classifier is None:
        classifier = load_model()
    cleaned = [preprocess(t) for t in texts]
    return classifier(cleaned)

if __name__ == "__main__":
    examples = [
        "I absolutely love this product, amazing experience!",
        "Worst customer service I have ever seen.",
        "@AcmeSupport the package arrived on Tuesday, see https://example.com",
    ]
    for text, result in zip(examples, analyze(examples)):
        print(f"{result['label']:>8} ({result['score']:.2f}) — {text}")