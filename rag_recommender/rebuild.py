"""
Script to rebuild the embeddings and search index.
This should be run after updating the assessment data.
"""
import logging
from pathlib import Path
import numpy as np
import faiss
import pickle

from rag_recommender.modules.ingestion import load_assessments
from rag_recommender.modules.generate_embeddings import prepare_texts, generate_embedding

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
EMBEDDINGS_PATH = Path("embeddings.npy")
TEXTS_PATH = Path("vector_texts.pkl")
INDEX_PATH = Path("vector.index")

def rebuild_embeddings_and_index():
    """
    Rebuild embeddings and search index from the assessment data.
    """
    # Step 1: Load assessment data
    logging.info("Loading assessment data...")
    df = load_assessments()
    
    # Step 2: Prepare texts
    logging.info("Preparing texts for embedding...")
    texts = prepare_texts(df)
    logging.info(f"Prepared {len(texts)} texts for embedding.")
    logging.info(f"Sample: {texts[0]}")
    
    # Step 3: Generate embeddings
    logging.info("Generating embeddings (this may take a while)...")
    embeddings = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            logging.info(f"Processing {i}/{len(texts)} texts...")
        embedding = generate_embedding(text)
        embeddings.append(embedding)
    
    # Step 4: Save embeddings and texts
    logging.info("Converting embeddings to numpy array...")
    embedding_matrix = np.array(embeddings).astype("float32")
    
    logging.info(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embedding_matrix)
    
    logging.info(f"Saving texts to {TEXTS_PATH}...")
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    
    # Step 5: Build and save FAISS index
    logging.info("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    
    logging.info(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))
    
    logging.info("Rebuild complete!")
    return len(texts)

if __name__ == "__main__":
    logging.info("Starting rebuild process...")
    count = rebuild_embeddings_and_index()
    logging.info(f"Successfully rebuilt embeddings and index with {count} assessments!") 