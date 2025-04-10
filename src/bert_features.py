from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np
import os
from data_loader import load_authorship_dataset


def extract_bert_embeddings(
    dataset_path: str = "./../data/raw/dataset_authorship",
    save_dir: str = "./../data/processed",
    model_name: str = "dbmdz/bert-base-turkish-cased",
    max_length: int = 256,
):
    """
    Extracts BERT embeddings from Turkish texts using mean pooling.

    Args:
        dataset_path (str): Path to the dataset folder.
        save_dir (str): Path to save the embeddings and labels.
        model_name (str): Pretrained BERT model name.
        max_length (int): Max sequence length for BERT tokenizer.

    Saves:
        - X_bert_embeddings.npy
        - y_bert.npy
    """

    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("📂 Loading dataset...")
    df, label_encoder = load_authorship_dataset(dataset_path)
    texts = df["text"].tolist()

    embeddings = []
    print("🚀 Generating embeddings...")
    with torch.no_grad():
        for text in tqdm(texts, desc="Generating BERT embeddings"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            mean_pooled = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(mean_pooled)

    X_bert = np.vstack(embeddings)
    y = df["label"].values

    print(f"✅ Embeddings shape: {X_bert.shape}")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_bert_embeddings.npy"), X_bert)
    np.save(os.path.join(save_dir, "y_bert.npy"), y)
    print(f"💾 Saved to {save_dir}")
