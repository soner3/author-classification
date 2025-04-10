from data_loader import load_authorship_dataset

def test_load_authorship_dataset():
    df, le = load_authorship_dataset("data/raw/dataset_authorship", encode_labels=True)

    # Basic structure checks
    assert not df.empty, "DataFrame is empty"
    assert 'author' in df.columns and 'text' in df.columns and 'label' in df.columns

    # Check that exactly 30 unique authors are loaded
    expected_num_authors = 30
    actual_num_authors = df['author'].nunique()
    assert actual_num_authors == expected_num_authors, f"Expected {expected_num_authors} authors, but got {actual_num_authors}"

    # Check that label encoding matches
    assert df['label'].nunique() == expected_num_authors, "Label count does not match number of authors"
