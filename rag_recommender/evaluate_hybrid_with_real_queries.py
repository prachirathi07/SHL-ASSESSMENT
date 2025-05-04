"""
Evaluation script for the Hybrid SHL Assessment Recommendation System.
This script evaluates the performance of the hybrid approach on real test queries.
"""
import logging
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any, Tuple

from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf, build_tfidf_model
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
from rag_recommender.modules.evaluator import recall_at_k, average_precision_at_k

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Real test queries with ground truth assessments
REAL_TEST_QUERIES = [
    ("I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.", [
        "Automata - Fix (New)", 
        "Core Java (Entry Level) (New)", 
        "Java 8 (New)", 
        "Core Java (Advanced Level) (New)", 
        "Agile Software Development", 
        "Technology Professional 8.0 Job Focused Assessment", 
        "Computer Science (New)"
    ]),
    ("I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options", [
        "Entry level Sales 7.1 (International)", 
        "Entry Level Sales Sift Out 7.1", 
        "Entry Level Sales Solution", 
        "Sales Representative Solution", 
        "Sales Support Specialist Solution", 
        "Technical Sales Associate Solution", 
        "SVAR - Spoken English (Indian Accent) (New)", 
        "Sales & Service Phone Solution", 
        "Sales & Service Phone Simulation", 
        "English Comprehension (New)"
    ]),
    ("I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour", [
        "Motivation Questionnaire MQM5", 
        "Global Skills Assessment", 
        "Graduate 8.0 Job Focused Assessment"
    ]),
    ("Content Writer required, expert in English and SEO.", [
        "Drupal (New)", 
        "Search Engine Optimization (New)", 
        "Administrative Professional - Short Form", 
        "Entry Level Sales Sift Out 7.1", 
        "General Entry Level – Data Entry 7.0 Solution"
    ]),
    ("Find me 1 hour long assessment for a QA Engineer with experience in JavaScript, CSS, HTML, Selenium WebDriver and SQL server", [
        "Automata Selenium", 
        "Automata - Fix (New)", 
        "Automata Front End", 
        "JavaScript (New)", 
        "HTML/CSS (New)", 
        "HTML5 (New)", 
        "CSS3 (New)", 
        "Selenium (New)", 
        "SQL Server (New)", 
        "Automata - SQL (New)", 
        "Manual Testing (New)"
    ]),
    ("ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long", [
        "Administrative Professional - Short Form", 
        "Verify - Numerical Ability", 
        "Financial Professional - Short Form", 
        "Bank Administrative Assistant - Short Form", 
        "General Entry Level – Data Entry 7.0 Solution", 
        "Basic Computer Literacy (Windows 10) (New)"
    ]),
    ("Looking for a radio station programming manager with excellent communication skills, ability to work with sales teams, and people management experience", [
        "Verify - Verbal Ability - Next Generation", 
        "SHL Verify Interactive - Inductive Reasoning", 
        "Occupational Personality Questionnaire OPQ32r"
    ])
]

def extract_assessment_name(result: str) -> str:
    """Extract just the assessment name from a result string."""
    if "|" in result:
        return result.split("|")[0].strip()
    return result

def evaluate_recommender_hybrid_real(detailed: bool = False):
    """
    Evaluate the hybrid recommender on real test queries.
    
    Args:
        detailed: Whether to show detailed results for each query
    """
    # First, rebuild the TF-IDF model to ensure we're using the latest version
    logging.info("First, rebuilding the TF-IDF model to ensure we're using the latest version...")
    build_tfidf_model()
    
    # Prepare storage for results
    results_df = pd.DataFrame(columns=[
        'Query', 'Relevant Count', 'Found Count', 
        'Recall@1', 'AP@1',
        'Recall@3', 'AP@3',
        'Recall@5', 'AP@5',
        'Recall@10', 'AP@10'
    ])
    
    # Track overall metrics
    recall_at_1_sum = 0
    recall_at_3_sum = 0
    recall_at_5_sum = 0
    recall_at_10_sum = 0
    ap_at_1_sum = 0
    ap_at_3_sum = 0
    ap_at_5_sum = 0
    ap_at_10_sum = 0
    
    # Evaluate each test query
    for i, (query, relevant_assessments) in enumerate(REAL_TEST_QUERIES, 1):
        truncated_query = f"{query[:40]}..." if len(query) > 40 else query
        logging.info(f"\nEvaluating query {i}: {truncated_query}")
        
        # Get recommendations using the hybrid approach
        results = search_assessments_hybrid(query, top_k=10)
        
        # Extract assessment names for comparison with ground truth
        result_names = [extract_assessment_name(result) for result in results]
        
        # Calculate metrics for this query
        found_assessments = [name for name in result_names if name in relevant_assessments]
        found_count = len(found_assessments)
        
        # Calculate evaluation metrics
        recall_at_1_val = recall_at_k(result_names, relevant_assessments, 1)
        recall_at_3_val = recall_at_k(result_names, relevant_assessments, 3)
        recall_at_5_val = recall_at_k(result_names, relevant_assessments, 5)
        recall_at_10_val = recall_at_k(result_names, relevant_assessments, 10)
        
        ap_at_1_val = average_precision_at_k(result_names, relevant_assessments, 1)
        ap_at_3_val = average_precision_at_k(result_names, relevant_assessments, 3)
        ap_at_5_val = average_precision_at_k(result_names, relevant_assessments, 5)
        ap_at_10_val = average_precision_at_k(result_names, relevant_assessments, 10)
        
        # Add to running totals
        recall_at_1_sum += recall_at_1_val
        recall_at_3_sum += recall_at_3_val
        recall_at_5_sum += recall_at_5_val
        recall_at_10_sum += recall_at_10_val
        ap_at_1_sum += ap_at_1_val
        ap_at_3_sum += ap_at_3_val
        ap_at_5_sum += ap_at_5_val
        ap_at_10_sum += ap_at_10_val
        
        # Store the results for this query
        row_data = {
            'Query': query,
            'Relevant Count': len(relevant_assessments),
            'Found Count': found_count,
            'Recall@1': recall_at_1_val,
            'AP@1': ap_at_1_val,
            'Recall@3': recall_at_3_val,
            'AP@3': ap_at_3_val,
            'Recall@5': recall_at_5_val,
            'AP@5': ap_at_5_val,
            'Recall@10': recall_at_10_val,
            'AP@10': ap_at_10_val
        }
        results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Log metrics for this query
        logging.info(f"  Query: {query}")
        logging.info(f"  Relevant count: {len(relevant_assessments)}")
        logging.info(f"  Found {found_count} out of {len(relevant_assessments)} relevant assessments")
        logging.info(f"  Recall@1: {recall_at_1_val:.4f}, AP@1: {ap_at_1_val:.4f}")
        logging.info(f"  Recall@3: {recall_at_3_val:.4f}, AP@3: {ap_at_3_val:.4f}")
        logging.info(f"  Recall@5: {recall_at_5_val:.4f}, AP@5: {ap_at_5_val:.4f}")
        logging.info(f"  Recall@10: {recall_at_10_val:.4f}, AP@10: {ap_at_10_val:.4f}")
        
        # Show detailed recommendations if requested
        if detailed:
            logging.info("\n  Top 5 recommendations:")
            for j, result in enumerate(results[:5], 1):
                name = extract_assessment_name(result)
                relevant_marker = "✓" if name in relevant_assessments else "✗"
                logging.info(f"    {j}. {relevant_marker} {name}")
            
            # Show relevant assessments that weren't found in the top 10
            not_found = set(relevant_assessments) - set(found_assessments)
            if not_found:
                logging.info("\n  Relevant assessments not found in top 10:")
                for j, name in enumerate(not_found, 1):
                    logging.info(f"    {j}. {name}")
            
    # Calculate and log overall metrics
    num_queries = len(REAL_TEST_QUERIES)
    logging.info("=" * 50)
    logging.info("EVALUATION SUMMARY (HYBRID APPROACH):")
    logging.info("=" * 50)
    logging.info("\nOverall Metrics:")
    
    mean_recall_at_1 = recall_at_1_sum / num_queries
    mean_recall_at_3 = recall_at_3_sum / num_queries
    mean_recall_at_5 = recall_at_5_sum / num_queries
    mean_recall_at_10 = recall_at_10_sum / num_queries
    mean_ap_at_1 = ap_at_1_sum / num_queries
    mean_ap_at_3 = ap_at_3_sum / num_queries
    mean_ap_at_5 = ap_at_5_sum / num_queries
    mean_ap_at_10 = ap_at_10_sum / num_queries
    
    logging.info(f"Mean Recall@1: {mean_recall_at_1:.4f}")
    logging.info(f"MAP@1: {mean_ap_at_1:.4f}")
    logging.info(f"Mean Recall@3: {mean_recall_at_3:.4f}")
    logging.info(f"MAP@3: {mean_ap_at_3:.4f}")
    logging.info(f"Mean Recall@5: {mean_recall_at_5:.4f}")
    logging.info(f"MAP@5: {mean_ap_at_5:.4f}")
    logging.info(f"Mean Recall@10: {mean_recall_at_10:.4f}")
    logging.info(f"MAP@10: {mean_ap_at_10:.4f}")
    
    # Per-query summary
    logging.info("\nPer-Query Recall Summary:")
    for i, row in results_df.iterrows():
        query_text = row['Query']
        truncated_query = f"{query_text[:30]}..." if len(query_text) > 30 else query_text
        logging.info(f"Query {i+1}: Found {row['Found Count']}/{row['Relevant Count']} relevant assessments, "
                    f"Recall@1: {row['Recall@1']:.2f}, "
                    f"Recall@3: {row['Recall@3']:.2f}, "
                    f"Recall@5: {row['Recall@5']:.2f}, "
                    f"Recall@10: {row['Recall@10']:.2f}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the hybrid SHL recommendation system")
    parser.add_argument('--detailed', action='store_true', 
                        help='Show detailed results for each query')
    args = parser.parse_args()
    
    evaluate_recommender_hybrid_real(detailed=args.detailed) 