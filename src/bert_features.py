import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from data_loader import load_authorship_dataset
from logger import get_logger

# Init logger
logger = get_logger(__name__)


def extract_bert_embeddings(
    dataset_path: str = "data/raw/dataset_authorship",
    save_dir: str = "data/processed",
    model_name: str = "dbmdz/bert-base-turkish-cased",
    max_length: int = 256,
    batch_size: int = 16,
):
    """
    Extracts BERT embeddings using mean pooling in batches.

    Args:
        dataset_path (str): Path to dataset directory.
        save_dir (str): Directory to save outputs.
        model_name (str): HuggingFace model name.
        max_length (int): Max sequence length for tokenizer.
        batch_size (int): Number of documents per batch.

    Saves:
        - X_bert_embeddings.npy
        - y_bert.npy
    """

    logger.info("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"📦 Using device: {device}")

    logger.info("📂 Loading dataset...")
    df, _ = load_authorship_dataset(dataset_path)
    texts = df["text"].tolist()
    y = df["label"].values

    logger.info("🚀 Generating embeddings in batches...")
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            mean_pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

    X_bert = np.vstack(embeddings)
    logger.info(f"✅ Embeddings shape: {X_bert.shape}")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_bert_embeddings.npy"), X_bert)
    np.save(os.path.join(save_dir, "y_bert.npy"), y)

    logger.info(f"💾 Embeddings and labels saved to: {save_dir}")
