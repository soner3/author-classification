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

    Args:
        texts (list): List of input text documents.
        ngram_range (tuple): N-gram range (e.g., (1, 1) for unigrams).
        analyzer (str): 'word' or 'char'.
        max_features (int): Maximum number of features to keep.
        stopword_list (list or None): Custom list of stopwords.

    Returns:
        Tuple of:
            - csr_matrix: TF-IDF matrix in sparse format
            - CountVectorizer: The fitted vectorizer
            - np.ndarray: Array of IDF values
    """

    # Ensure stopword list is in the correct format
    if stopword_list is not None and not isinstance(stopword_list, list):
        stopword_list = list(stopword_list)

    # 1. Compute raw term frequencies
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stopword_list,
        lowercase=True,
        max_features=max_features,
    )
    X_counts = vectorizer.fit_transform(texts).astype(np.float32)

    # 2. Apply Double Normalization 0.5
    X_counts = X_counts.tocsr()
    max_tf = X_counts.max(axis=1).toarray().ravel()
    max_tf[max_tf == 0] = 1  # Avoid division by zero

    X_coo = X_counts.tocoo(copy=False)
    X_coo.data = 0.5 + 0.5 * (X_coo.data / max_tf[X_coo.row])
    X_counts = X_coo.tocsr()

    # 3. Compute smoothed IDF
    df = X_counts.getnnz(axis=0)  # document frequency for each term
    n_docs = X_counts.shape[0]
    idf = np.log((1 + n_docs) / (1 + df)) + 1  # smoothed IDF

    # 4. Apply TF × IDF weighting
    X_tfidf = X_counts.tocsc(copy=False)
    X_tfidf.data *= idf[X_tfidf.indices]
    X_tfidf = X_tfidf.tocsr()

    return X_tfidf, vectorizer, idf
