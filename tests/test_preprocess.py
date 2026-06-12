"""
UNIT TEST deterministici per la funzione preprocess() di src/sentiment.py.
"""
from src.sentiment import preprocess

def test_mention_is_anonymized():
    """Una menzione (@nomeutente) deve essere sostituita dal placeholder @user,
    perche' il modello in addestramento ha visto solo menzioni anonimizzate."""
    assert preprocess("@marco23 stai esagerando") == "@user stai esagerando"

def test_url_is_normalized():
    """Un URL deve collassare nel placeholder 'http': l'indirizzo specifico
    non porta informazione di sentiment ed e' assente dai dati di training."""
    assert preprocess("guarda qui https://example.com/offerta") == "guarda qui http"

def test_clean_text_is_unchanged():
    """Testo senza menzioni ne' URL deve uscire IDENTICO: il preprocessing
    non deve mai danneggiare input che non richiede normalizzazione."""
    text = "I love this product"
    assert preprocess(text) == text

def test_lone_at_symbol_is_preserved():
    """Edge case: una '@' isolata non e' una menzione e non va trasformata.
    Protegge il controllo len(t) > 1 presente nel codice: se qualcuno lo
    rimuovesse, questo test diventerebbe rosso."""
    assert preprocess("ci vediamo @ casa") == "ci vediamo @ casa"

def test_mixed_mention_and_url():
    """Menzione e URL nella stessa frase: le due sostituzioni devono
    convivere senza interferire l'una con l'altra."""
    assert preprocess("@AcmeSupport check https://acme.com now") == "@user check http now"