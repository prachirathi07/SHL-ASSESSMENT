"""
Evaluation script for the SHL Assessment Recommendation System.
This script calculates Mean Recall@K and MAP@K metrics using the provided test dataset.
"""
import logging
from rag_recommender.modules.rag_pipeline import search_assessments
from rag_recommender.modules.evaluator import evaluate_test_set
from rag_recommender.data.test_queries import TEST_QUERIES

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def run_evaluation(k=3):
    """
    Run evaluation on the test dataset and print detailed results.
    """
    logging.info(f"Starting evaluation with k={k}...")
    
    # Evaluate using the test queries
    metrics = evaluate_test_set(search_assessments, TEST_QUERIES, k)
    
    # Print detailed results
    logging.info(f"Evaluation Results:")
    logging.info(f"Mean Recall@{k}: {metrics['mean_recall_at_k']:.4f}")
    logging.info(f"MAP@{k}: {metrics['map_at_k']:.4f}")
    
    # Print per-query results for detailed analysis
    logging.info("\nPer-query Results:")
    for i, test_case in enumerate(TEST_QUERIES):
        query = test_case['query']
        relevant = test_case['relevant']
        try:
            recommended = search_assessments(query)
            logging.info(f"\nQuery {i+1}: {query[:50]}...")
            logging.info(f"  Recommended (top {k}):")
            for j, rec in enumerate(recommended[:k]):
                logging.info(f"    {j+1}. {rec}")
            logging.info(f"  Relevant assessments ({len(relevant)}):")
            for j, rel in enumerate(relevant):
                logging.info(f"    {j+1}. {rel}")
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
    
    return metrics

if __name__ == "__main__":
    # Run evaluation with k=3 (as specified in requirements)
    metrics = run_evaluation(k=3)
    
    # Also run evaluation with k=10 to see full performance
    metrics_10 = run_evaluation(k=10) 