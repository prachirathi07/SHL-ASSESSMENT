"""
Evaluate the performance of the hybrid recommendation system.
This script tests recommendation quality on a set of predefined test queries.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def extract_assessment_name(result: str) -> str:
    """Extract just the assessment name from a result string."""
    if "|" in result:
        return result.split("|")[0].strip()
    return result

def compute_precision_at_k(recommended: List[str], expected: List[str], k: int) -> float:
    """
    Compute Precision@k for a single recommendation.
    
    Args:
        recommended: List of recommended assessment names
        expected: List of expected/relevant assessment names
        k: Number of top recommendations to consider
        
    Returns:
        Precision@k value between 0 and 1
    """
    if not recommended[:k]:
        return 0.0  # If no recommendations, precision is 0
        
    # Extract just the assessment names from the recommendations
    rec_names = [extract_assessment_name(rec) for rec in recommended[:k]]
    
    # Count how many of the top-k recommendations are relevant
    relevant = sum(1 for rec in rec_names if any(exp.lower() in rec.lower() for exp in expected))
    
    # Precision@k = number of relevant items in top-k / k
    return relevant / k

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

def compute_recall_at_k(recommended: List[str], expected: List[str], k: int) -> float:
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

def evaluate_system():
    """Evaluate the recommendation system using test queries."""
    results = {
        "query": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "precision@1": [],
        "precision@3": [],
        "precision@5": [],
        "precision@10": []
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
            
            # Compute recall and precision at different k values
            recall_1 = compute_recall_at_k(recommendations, expected, 1)
            recall_3 = compute_recall_at_k(recommendations, expected, 3)
            recall_5 = compute_recall_at_k(recommendations, expected, 5)
            recall_10 = compute_recall_at_k(recommendations, expected, 10)
            
            precision_1 = compute_precision_at_k(recommendations, expected, 1)
            precision_3 = compute_precision_at_k(recommendations, expected, 3)
            precision_5 = compute_precision_at_k(recommendations, expected, 5)
            precision_10 = compute_precision_at_k(recommendations, expected, 10)
            
            # Log metrics
            logging.info(f"Recall@1: {recall_1:.4f}")
            logging.info(f"Recall@3: {recall_3:.4f}")
            logging.info(f"Recall@5: {recall_5:.4f}")
            logging.info(f"Recall@10: {recall_10:.4f}")
            logging.info(f"Precision@1: {precision_1:.4f}")
            logging.info(f"Precision@3: {precision_3:.4f}")
            logging.info(f"Precision@5: {precision_5:.4f}")
            logging.info(f"Precision@10: {precision_10:.4f}")
            
            # Store results
            results["query"].append(query)
            results["recall@1"].append(recall_1)
            results["recall@3"].append(recall_3)
            results["recall@5"].append(recall_5)
            results["recall@10"].append(recall_10)
            results["precision@1"].append(precision_1)
            results["precision@3"].append(precision_3)
            results["precision@5"].append(precision_5)
            results["precision@10"].append(precision_10)
        except Exception as e:
            logging.error(f"Error evaluating query: {e}")
        
        logging.info("-" * 50)
    
    # Calculate mean metrics
    if results["recall@1"]:
        mean_recall_1 = np.mean(results["recall@1"])
        mean_recall_3 = np.mean(results["recall@3"])
        mean_recall_5 = np.mean(results["recall@5"])
        mean_recall_10 = np.mean(results["recall@10"])
        
        mean_precision_1 = np.mean(results["precision@1"])
        mean_precision_3 = np.mean(results["precision@3"])
        mean_precision_5 = np.mean(results["precision@5"])
        mean_precision_10 = np.mean(results["precision@10"])
        
        # Log summary
        logging.info("\n=== EVALUATION SUMMARY ===")
        logging.info(f"Number of test queries evaluated: {len(results['recall@1'])}")
        logging.info(f"Mean Recall@1: {mean_recall_1:.4f} ({mean_recall_1*100:.2f}%)")
        logging.info(f"Mean Recall@3: {mean_recall_3:.4f} ({mean_recall_3*100:.2f}%)")
        logging.info(f"Mean Recall@5: {mean_recall_5:.4f} ({mean_recall_5*100:.2f}%)")
        logging.info(f"Mean Recall@10: {mean_recall_10:.4f} ({mean_recall_10*100:.2f}%)")
        logging.info(f"Mean Precision@1: {mean_precision_1:.4f} ({mean_precision_1*100:.2f}%)")
        logging.info(f"Mean Precision@3: {mean_precision_3:.4f} ({mean_precision_3*100:.2f}%)")
        logging.info(f"Mean Precision@5: {mean_precision_5:.4f} ({mean_precision_5*100:.2f}%)")
        logging.info(f"Mean Precision@10: {mean_precision_10:.4f} ({mean_precision_10*100:.2f}%)")
    else:
        logging.error("No successful evaluations to report")
    
    return results

if __name__ == "__main__":
    logging.info("Starting evaluation of hybrid recommendation system...")
    results = evaluate_system()
    logging.info("Evaluation complete.") 