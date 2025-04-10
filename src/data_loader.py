import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_authorship_dataset(
    dataset_path: str, encode_labels: bool = True
) -> tuple[pd.DataFrame, LabelEncoder | None]:
    """
    Loads text files from directories (authors as folders) into a DataFrame.

    Args:
        dataset_path (str): Path to root folder (each subfolder = one author).
        encode_labels (bool): If True, encodes author names into numeric labels.

    Returns:
        Tuple:
            - pd.DataFrame with ['author', 'text', 'label'] if encode_labels=True
            - LabelEncoder instance (or None if encode_labels=False)
    """
    data = []

    for author in os.listdir(dataset_path):
        author_path = os.path.join(dataset_path, author)
        if os.path.isdir(author_path):
            for filename in os.listdir(author_path):
                if filename.endswith(".txt"):
                    try:
                        with open(
                            os.path.join(author_path, filename), "r", encoding="utf-8"
                        ) as f:
                            text = f.read()
                            data.append({"author": author, "text": text})
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    df = pd.DataFrame(data).dropna()

    if encode_labels:
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["author"])
        return df, le
    else:
        return df, None
