"""
INTEGRATION TEST per la pipeline di sentiment analysis.

Verificano che il codice funzioni correttamente insieme al componente esterno 
(il modello RoBERTa scaricato dal Hub di Hugging Face).
"""
import pytest
from src.sentiment import load_model, analyze

@pytest.fixture(scope="module")
def classifier():
    """Carica il modello una sola volta per tutto il file di test."""
    return load_model()

def test_obviously_positive(classifier):
    """Una frase inequivocabilmente positiva deve essere classificata
    'positive'. Se fallisce, e' rotto qualcosa di strutturale, non una
    sfumatura del modello."""
    result = analyze(["I absolutely love this, best purchase ever!"], classifier)[0]
    assert result["label"].lower() == "positive"

def test_obviously_negative(classifier):
    """Una frase inequivocabilmente negativa deve essere classificata 'negative'."""
    result = analyze(["Horrible experience, total waste of money."], classifier)[0]
    assert result["label"].lower() == "negative"

def test_neutral_factual_statement(classifier):
    """Un'affermazione puramente fattuale deve risultare 'neutral',"""
    result = analyze(["The package was delivered on Tuesday."], classifier)[0]
    assert result["label"].lower() == "neutral"

def test_score_is_valid_probability(classifier):
    """Contratto sul FORMATO dell'output: lo score deve essere una
    probabilita' in [0, 1]. Protegge i consumatori a valle dello score
    (es. il monitoraggio della confidenza media) da cambi di formato
    della libreria (es. logit grezzi al posto di probabilita')."""
    result = analyze(["What a fantastic day!"], classifier)[0]
    assert 0.0 <= result["score"] <= 1.0

def test_batch_returns_one_result_per_text(classifier):
    """Contratto di cardinalita': N testi in ingresso -> N risultati in
    uscita. Se un componente scartasse silenziosamente degli input, i
    risultati si disallineerebbero dai testi: ogni post riceverebbe il
    sentiment di un altro."""
    texts = ["Great!", "Awful!", "It is Tuesday."]
    results = analyze(texts, classifier)
    assert len(results) == len(texts)