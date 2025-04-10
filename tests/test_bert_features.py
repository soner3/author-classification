import shutil
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import torch
from bert_features import extract_bert_embeddings


@patch("bert_features.AutoTokenizer")
@patch("bert_features.AutoModel")
@patch("bert_features.load_authorship_dataset")
def test_extract_bert_embeddings_mocked(
    mock_load_dataset, mock_model_class, mock_tokenizer_class, tmp_path
):
    """
    Mocked test for extract_bert_embeddings().
    Ensures correct saving and dimensions without using real BERT.
    """

    # 1. Mock dataframe
    mock_df = pd.DataFrame(
        {"text": ["Mock text 1", "Another mock", "Final sentence"], "label": [0, 1, 2]}
    )
    mock_load_dataset.return_value = (mock_df, None)

    # 2. Mock tokenizer (callable)
    mock_tokenizer = MagicMock(
        return_value={
            "input_ids": torch.zeros((1, 64), dtype=torch.long),
            "attention_mask": torch.ones((1, 64), dtype=torch.long),
        }
    )
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # 3. Mock model output
    dummy_hidden = torch.rand((1, 64, 768))  # B x T x H
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = dummy_hidden

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = None
    mock_model.return_value = mock_outputs
    mock_model.__call__ = MagicMock(return_value=mock_outputs)
    mock_model_class.from_pretrained.return_value = mock_model

    # 4. Run function
    extract_bert_embeddings(dataset_path="unused", save_dir=tmp_path, max_length=64)

    # 5. Check results
    X = np.load(tmp_path / "X_bert_embeddings.npy")
    y = np.load(tmp_path / "y_bert.npy")

    assert X.shape == (3, 768), "Embedding matrix should have shape (3, 768)"
    assert y.shape == (3,), "Label array should have shape (3,)"
    assert y.tolist() == [0, 1, 2], "Labels should match expected values"
    shutil.rmtree(tmp_path)
