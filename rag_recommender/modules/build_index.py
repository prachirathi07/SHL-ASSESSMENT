import pickle
import logging
from pathlib import Path

import numpy as np
import faiss

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
EMBEDDINGS_PATH = Path("embeddings.npy")
TEXTS_PATH = Path("vector_texts.pkl")
INDEX_PATH = Path("vector.index")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, texts):
    faiss.write_index(index, str(INDEX_PATH))
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    logging.info("FAISS index and texts saved.")


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)

    index = build_index(embeddings)
    save_index(index, texts)
