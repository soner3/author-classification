from data_loader import load_authorship_dataset


def test_load_authorship_dataset():
    """
    Unit test for the `load_authorship_dataset` function.

    This test verifies that:
    - The dataset is successfully loaded into a non-empty DataFrame
    - The DataFrame contains the required columns: 'author', 'text', and 'label'
    - Exactly 30 unique authors are loaded (based on the expected dataset structure)
    - The label encoder correctly encodes each author into a unique numeric label
    """

    # Load dataset with numeric label encoding
    df, le = load_authorship_dataset("data/raw/dataset_authorship", encode_labels=True)

    # Basic structure checks
    assert not df.empty, "DataFrame is empty"
    assert "author" in df.columns, "'author' column is missing"
    assert "text" in df.columns, "'text' column is missing"
    assert "label" in df.columns, "'label' column is missing"

    # Check number of unique authors
    expected_num_authors = 30
    actual_num_authors = df["author"].nunique()
    assert (
        actual_num_authors == expected_num_authors
    ), f"Expected {expected_num_authors} authors, but got {actual_num_authors}"

    # Check that label encoding corresponds to number of authors
    actual_label_count = df["label"].nunique()
    assert (
        actual_label_count == expected_num_authors
    ), f"Label count mismatch: expected {expected_num_authors}, got {actual_label_count}"
