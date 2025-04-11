import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional
from logger import get_logger

# Setup logger (verwende __name__ für Modul-spezifisches Logging)
logger = get_logger(__name__)


def load_authorship_dataset(
    dataset_path: str, encode_labels: bool = True
) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Loads a labeled authorship dataset from directory structure into a DataFrame.

    Args:
        dataset_path (str): Path to dataset root (subfolders = authors, .txt files = samples).
        encode_labels (bool): Whether to encode author names to numeric labels.

    Returns:
        Tuple:
            - DataFrame with columns ['author', 'text', 'label'] (if encode_labels=True)
            - Fitted LabelEncoder (or None if not used)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    data = []
    authors = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    logger.info(f"📂 Found {len(authors)} author folders.")

    for author in authors:
        author_path = os.path.join(dataset_path, author)
        files = [f for f in os.listdir(author_path) if f.endswith(".txt")]

        for fname in files:
            fpath = os.path.join(author_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    if content:
                        data.append({"author": author, "text": content})
            except Exception as e:
                logger.warning(f"⚠️ Could not read file {fpath}: {e}")

    if not data:
        raise ValueError("No valid text files found in dataset.")

    df = pd.DataFrame(data).dropna()

    le = None
    if encode_labels:
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["author"])
        logger.info(f"🔢 Encoded {df['label'].nunique()} unique authors.")

    logger.info(f"✅ Loaded {len(df)} documents from dataset.")
    return df, le
