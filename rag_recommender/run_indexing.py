# run_indexing.py
from rag_recommender.modules.generate_embeddings import build_and_save_index
from modules.ingestion import load_assessments

if __name__ == "__main__":
    df = load_assessments()
    build_and_save_index(df)
