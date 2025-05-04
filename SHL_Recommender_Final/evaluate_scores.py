"""
Standalone script to evaluate the hybrid recommender system.
"""
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def extract_assessment_name(result_str):
    """Extract just the assessment name from a result string."""
    if "|" in result_str:
        return result_str.split("|")[0].strip()
    return result_str

# Test queries with their expected assessments
TEST_QUERIES = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "expected_assessments": [
            "Core Java (Entry Level) (New)",
            "Core Java (Advanced Level) (New)",
            "Java 8 (New)",
            "Enterprise Java Beans (New)"
        ]
    },
    {
        "query": "Need assessments for data science position requiring Python and machine learning skills.",
        "expected_assessments": [
            "Data Science (New)",
            "Python (New)",
            "Software Development Fundamentals - Python (New)",
            "Computer Science (New)",
            "SHL Verify Interactive - Inductive Reasoning"
        ]
    },
    {
        "query": "Looking for a content writer with SEO knowledge and English writing skills.",
        "expected_assessments": [
            "Search Engine Optimization (New)",
            "English Comprehension (New)",
            "Drupal (New)"
        ]
    },
    {
        "query": "We need to assess candidates for a sales position. The test should take less than 30 minutes.",
        "expected_assessments": [
            "Entry Level Sales Solution",
            "Sales & Service Phone Solution",
            "Entry level Sales 7.1 (International)",
            "Sales Representative Solution"
        ]
    },
    {
        "query": "QA engineer with Selenium testing experience.",
        "expected_assessments": [
            "Automata Selenium",
            "Selenium (New)",
            "Manual Testing (New)",
            "Quality Center (New)"
        ]
    }
]

def compute_recall_at_k(recommended, expected, k):
    """
    Compute Recall@k for a single recommendation.
    
    Args:
        recommended: List of recommended assessment names
        expected: List of expected/relevant assessment names
        k: Number of top recommendations to consider
        
    Returns:
        Recall@k value between 0 and 1
    """
    if not expected:
        return 1.0  # If no expected assessments, consider a perfect score
        
    # Extract just the assessment names from the recommendations
    rec_names = [extract_assessment_name(rec) for rec in recommended[:k]]
    
    # Count how many expected assessments appear in the top-k recommendations
    found = sum(1 for exp in expected if any(exp.lower() in rec.lower() for rec in rec_names))
    
    # Recall@k = number of relevant items found in top-k / total number of relevant items
    return found / len(expected)

def main():
    """Run the evaluation and print scores."""
    try:
        from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
        
        results = {
            "query": [],
            "recall@1": [],
            "recall@3": [],
            "recall@5": [],
            "recall@10": []
        }
        
        for test_case in TEST_QUERIES:
            query = test_case["query"]
            expected = test_case["expected_assessments"]
            
            logging.info(f"Evaluating query: {query}")
            logging.info(f"Expected assessments: {expected}")
            
            try:
                # Get recommendations
                recommendations = search_assessments_hybrid(query, top_k=10)
                
                # Log top 5 recommendations
                logging.info("Top 5 recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    logging.info(f"{i}. {rec}")
                
                # Compute recall at different k values
                recall_1 = compute_recall_at_k(recommendations, expected, 1)
                recall_3 = compute_recall_at_k(recommendations, expected, 3)
                recall_5 = compute_recall_at_k(recommendations, expected, 5)
                recall_10 = compute_recall_at_k(recommendations, expected, 10)
                
                # Log recall metrics
                logging.info(f"Recall@1: {recall_1:.4f}")
                logging.info(f"Recall@3: {recall_3:.4f}")
                logging.info(f"Recall@5: {recall_5:.4f}")
                logging.info(f"Recall@10: {recall_10:.4f}")
                
                # Store results
                results["query"].append(query)
                results["recall@1"].append(recall_1)
                results["recall@3"].append(recall_3)
                results["recall@5"].append(recall_5)
                results["recall@10"].append(recall_10)
            except Exception as e:
                logging.error(f"Error evaluating query: {e}")
            
            logging.info("-" * 50)
        
        # Calculate mean metrics
        if results["recall@1"]:
            mean_recall_1 = np.mean(results["recall@1"])
            mean_recall_3 = np.mean(results["recall@3"])
            mean_recall_5 = np.mean(results["recall@5"])
            mean_recall_10 = np.mean(results["recall@10"])
            
            # Log summary
            logging.info("\n=== EVALUATION SUMMARY ===")
            logging.info(f"Number of test queries evaluated: {len(results['recall@1'])}")
            logging.info(f"Mean Recall@1: {mean_recall_1:.4f} ({mean_recall_1*100:.2f}%)")
            logging.info(f"Mean Recall@3: {mean_recall_3:.4f} ({mean_recall_3*100:.2f}%)")
            logging.info(f"Mean Recall@5: {mean_recall_5:.4f} ({mean_recall_5*100:.2f}%)")
            logging.info(f"Mean Recall@10: {mean_recall_10:.4f} ({mean_recall_10*100:.2f}%)")
        else:
            logging.error("No successful evaluations to report")
            
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Make sure you're running this script from the clean_shl_recommender directory")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    logging.info("Starting evaluation of hybrid recommendation system...")
    sys.exit(main()) 