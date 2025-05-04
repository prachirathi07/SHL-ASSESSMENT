"""
Setup script for the SHL Assessment Recommendation System.
This script ensures all necessary models are built before running the system.
"""
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def main():
    """Main setup function to prepare the recommendation system."""
    logging.info("Setting up the SHL Assessment Recommendation System...")
    
    # Check if models exist
    model_files = [
        "tfidf_model.pkl",
        "tfidf_matrix.npy",
        "tfidf_texts.pkl",
        "assessment_df.pkl"
    ]
    
    missing_files = [f for f in model_files if not Path(f).exists()]
    
    if missing_files:
        logging.info(f"Missing model files: {', '.join(missing_files)}")
        logging.info("Building TF-IDF model...")
        
        # Import and run the build_tfidf module
        try:
            from rag_recommender.build_tfidf import build_tfidf_model
            build_tfidf_model()
            logging.info("TF-IDF model built successfully.")
        except Exception as e:
            logging.error(f"Error building TF-IDF model: {e}")
            return False
    else:
        logging.info("All model files exist. Setup complete.")
    
    logging.info("Testing recommendation system...")
    try:
        from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
        results = search_assessments_hybrid("Java developer", top_k=1)
        logging.info(f"Test recommendation: {results[0]}")
        logging.info("Recommendation system is working correctly.")
    except Exception as e:
        logging.error(f"Error testing recommendation system: {e}")
        return False
    
    logging.info("Setup complete! You can now run the system with 'python run.py'")
    return True

if __name__ == "__main__":
    main() 