import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from bert_features import extract_bert_embeddings


@patch("bert_features.AutoTokenizer")
@patch("bert_features.AutoModel")
@patch("bert_features.load_authorship_dataset")
def test_extract_bert_embeddings_mocked(
    mock_load_dataset, mock_model_class, mock_tokenizer_class, tmp_path
):
    """
    Mocked test for extract_bert_embeddings() with batch processing.
    Verifies correct shapes and saved outputs.
    """

    # 1. Create mock dataset
    mock_df = pd.DataFrame(
        {"text": ["Mock text 1", "Another mock", "Final sentence"], "label": [0, 1, 2]}
    )
    mock_load_dataset.return_value = (mock_df, None)

    # 2. Mock tokenizer (returns tensor batch of 3 samples)
    mock_tokenizer = MagicMock()
    mock_tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": torch.zeros((len(texts), 64), dtype=torch.long),
        "attention_mask": torch.ones((len(texts), 64), dtype=torch.long),
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # 3. Mock model (returns batch hidden states)
    def mock_model_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        dummy_hidden = torch.rand((batch_size, 64, 768))  # [B, T, H]
        return MagicMock(last_hidden_state=dummy_hidden)

    mock_model = MagicMock(side_effect=mock_model_forward)
    mock_model.eval.return_value = None
    mock_model.to.return_value = None
    mock_model_class.from_pretrained.return_value = mock_model

    # 4. Run embedding extraction (with batch_size=3)
    extract_bert_embeddings(
        dataset_path="unused", save_dir=tmp_path, max_length=64, batch_size=3
    )

    # 5. Load and validate output
    X = np.load(tmp_path / "X_bert_embeddings.npy")
    y = np.load(tmp_path / "y_bert.npy")

    assert X.shape == (3, 768), "Embedding matrix shape mismatch"
    assert y.shape == (3,), "Label vector shape mismatch"
    assert y.tolist() == [0, 1, 2], "Label content mismatch"
