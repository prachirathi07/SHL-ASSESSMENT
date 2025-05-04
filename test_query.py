"""
Test script to check if the SHL Assessment recommendation system is working properly.
This script takes a query and displays the top 5 recommendations.
"""
import logging
from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def test_query(query: str, top_k: int = 5):
    """
    Test the recommendation system with a query.
    
    Args:
        query: The test query
        top_k: Number of recommendations to display
    """
    logging.info(f"Testing query: '{query}'")
    try:
        results = search_assessments_tfidf(query, top_k=top_k)
        
        logging.info(f"Top {top_k} recommendations:")
        for i, result in enumerate(results, 1):
            logging.info(f"{i}. {result}")
            
        return results
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Test query - you can change this to any query you want to test
    query = "I need a Java programmer with good team collaboration skills"
    test_query(query, top_k=5) 