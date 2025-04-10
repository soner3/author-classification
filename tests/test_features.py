import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from features import extract_tfidf_double_norm


@pytest.mark.parametrize(
    "analyzer, ngram_range",
    [
        ("word", (1, 1)),
        ("word", (2, 2)),
        ("word", (3, 3)),
        ("char", (2, 2)),
        ("char", (3, 3)),
    ],
)
@patch("data_loader.load_authorship_dataset")
def test_all_feature_configs_mocked(mock_load_dataset, analyzer, ngram_range):
    """
    Test TF-IDF extraction using mocked dataset and stopword logic.
    This avoids loading real files and speeds up testing.
    """

    # Mocked sample dataset (3 short texts, 2 authors)
    mock_texts = [
        "Bu test cümlesi üç kelimeden oluşur",
        "İkinci cümlede de daha fazla kelime var",
        "Üçüncü cümle test için yeterince uzun olsun",
    ]

    mock_labels = [0, 1, 0]

    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: {
        "text": mock_texts,
        "label": np.array(mock_labels),
    }[key]

    mock_load_dataset.return_value = (mock_df, None)

    # Stopwords for Turkish (minimal for test)
    stop_words = ["bir", "bu"] if analyzer == "word" else None

    # Run extraction
    X, vectorizer, idf = extract_tfidf_double_norm(
        texts=mock_texts,
        ngram_range=ngram_range,
        analyzer=analyzer,
        stopword_list=stop_words,
        max_features=100,
    )

    # Assertions
    assert X.shape[0] == len(mock_texts), "Row count mismatch"
    assert X.shape[1] <= 100, "Too many features"
    assert X.nnz > 0, "TF-IDF matrix is empty"
    assert len(idf) == X.shape[1], "IDF vector size mismatch"
