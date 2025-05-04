"""
Script to build the TF-IDF model for the SHL Assessment Recommendation System.
This script builds an alternative recommendation system that doesn't rely on external APIs.
"""
import logging
from rag_recommender.modules.tfidf_recommender import build_tfidf_model, search_assessments_tfidf

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def main():
    """
    Build the TF-IDF model and test it with some queries.
    """
    logging.info("Building TF-IDF model...")
    build_tfidf_model()
    
    # Test queries
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams",
        "I need a test for a sales position for new graduates",
        "Looking for a COO assessment focused on cultural fit",
        "Content Writer required, expert in English and SEO",
        "QA Engineer with JavaScript, CSS, HTML and SQL skills"
    ]
    
    logging.info("\nTesting with sample queries...")
    for query in test_queries:
        results = search_assessments_tfidf(query, top_k=3)
        
        logging.info(f"\nQuery: {query}")
        logging.info("Top 3 recommendations:")
        for i, result in enumerate(results, 1):
            logging.info(f"{i}. {result}")

if __name__ == "__main__":
    main() 