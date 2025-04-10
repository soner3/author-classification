from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix


def extract_tfidf_double_norm(
    texts: list[str],
    ngram_range: Tuple[int, int] = (1, 1),
    analyzer: str = "word",
    max_features: int = 10000,
    stopword_list: list[str] = None,
) -> Tuple[csr_matrix, CountVectorizer, np.ndarray]:
    """
    Extract TF-IDF matrix using Double Normalization 0.5 and smoothed IDF.

    Returns:
        - TF-IDF matrix (csr sparse format)
        - Fitted CountVectorizer
        - IDF vector (NumPy array)
    """

    # 1. Count term frequencies (raw f_{t,d}) using CountVectorizer
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stopword_list,
        lowercase=True,
        max_features=max_features,
    )
    # Ergebnis: sparse matrix (docs × terms)
    X_counts = vectorizer.fit_transform(texts).astype(np.float32)

    # 2. Double Normalization TF:
    # tf = 0.5 + 0.5 * (f_{t,d} / max_{t'} f_{t',d})
    max_tf = X_counts.max(axis=1).toarray().ravel()  # max term freq pro Dokument
    max_tf[max_tf == 0] = 1  # Vermeidung Division durch 0

    # Zugriff auf alle Nicht-Null-Einträge in der Matrix
    row_indices, col_indices = X_counts.nonzero()
    for i, j in zip(row_indices, col_indices):
        X_counts[i, j] = 0.5 + 0.5 * (
            X_counts[i, j] / max_tf[i]
        )  # Double Norm anwenden

    # 3. IDF berechnen:
    # IDF(t) = log((1 + n) / (1 + df(t))) + 1
    df = np.diff(X_counts.tocsc().indptr)  # df(t): in wie vielen Dokus Term t vorkommt
    n_docs = X_counts.shape[0]
    idf = np.log((1 + n_docs) / (1 + df)) + 1

    # 4. TF × IDF multiplizieren
    X_tfidf = X_counts.tocsc()
    X_tfidf.data *= idf[X_tfidf.indices]  # jede Zelle: tf(t,d) × idf(t)
    X_tfidf = X_tfidf.tocsr()

    return X_tfidf, vectorizer, idf
