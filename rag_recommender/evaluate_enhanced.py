"""
Enhanced evaluation script for the TF-IDF based SHL Assessment Recommendation System.
This script includes detailed query-by-query metrics and recalls assessments from ground truth.
"""
import logging
import pandas as pd
import numpy as np
from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf, build_tfidf_model, filter_assessments
from rag_recommender.modules.evaluator import evaluate_test_set, recall_at_k, average_precision_at_k
from rag_recommender.data.test_queries import TEST_QUERIES

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def extract_assessment_name(result: str) -> str:
    """Extract just the assessment name from a result string."""
    if "|" in result:
        return result.split("|")[0].strip()
    return result.strip()

def run_enhanced_evaluation(k_values=[1, 3, 5, 10]):
    """
    Run an enhanced evaluation with multiple k values and detailed per-query metrics.
    
    Args:
        k_values: List of k values to evaluate
    """
    # Rebuild the model to ensure we're using the latest version
    logging.info("First, rebuilding the TF-IDF model to ensure we're using the latest version...")
    build_tfidf_model()
    
    # Prepare results dataframe
    results_df = pd.DataFrame(columns=['Query', 'Relevant Count', 'Found Count'] + 
                             [f'Recall@{k}' for k in k_values] + 
                             [f'MAP@{k}' for k in k_values])
    
    # Overall metrics
    all_metrics = {}
    
    # Process each test query
    for i, test_case in enumerate(TEST_QUERIES):
        query = test_case['query']
        relevant = test_case['relevant']
        
        logging.info(f"\nEvaluating query {i+1}: {query[:50]}...")
        try:
            recommended = search_assessments_tfidf(query, top_k=max(k_values))
            recommended_names = [extract_assessment_name(rec) for rec in recommended]
            
            # Calculate metrics for each k
            recalls = [recall_at_k(recommended_names, relevant, k) for k in k_values]
            aps = [average_precision_at_k(recommended_names, relevant, k) for k in k_values]
            
            # Log detailed results
            logging.info(f"  Query: {query}")
            logging.info(f"  Relevant count: {len(relevant)}")
            found_count = sum(1 for r in relevant if any(r.lower() in rec.lower() for rec in recommended_names[:max(k_values)]))
            logging.info(f"  Found {found_count} out of {len(relevant)} relevant assessments")
            
            for j, k in enumerate(k_values):
                logging.info(f"  Recall@{k}: {recalls[j]:.4f}, AP@{k}: {aps[j]:.4f}")
            
            # Add row to results dataframe
            row_data = {
                'Query': query, 
                'Relevant Count': len(relevant), 
                'Found Count': found_count
            }
            for j, k in enumerate(k_values):
                row_data[f'Recall@{k}'] = recalls[j]
                row_data[f'MAP@{k}'] = aps[j]
            
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            
            # Print recommendations and relevant items
            logging.info(f"\n  Top {min(5, max(k_values))} recommendations:")
            for j, rec in enumerate(recommended_names[:min(5, max(k_values))]):
                is_relevant = any(rel.lower() in rec.lower() for rel in relevant)
                marker = "✓" if is_relevant else "✗"
                logging.info(f"    {j+1}. {marker} {rec}")
            
            logging.info(f"\n  Relevant assessments ({len(relevant)}):")
            for j, rel in enumerate(relevant):
                found = any(rel.lower() in rec.lower() for rec in recommended_names[:max(k_values)])
                marker = "✓" if found else "✗"
                logging.info(f"    {j+1}. {marker} {rel}")
            
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
    
    # Calculate overall metrics
    for k in k_values:
        recall_col = f'Recall@{k}'
        map_col = f'MAP@{k}'
        all_metrics[recall_col] = results_df[recall_col].mean()
        all_metrics[map_col] = results_df[map_col].mean()
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("EVALUATION SUMMARY")
    logging.info("="*50)
    
    logging.info("\nOverall Metrics:")
    for k in k_values:
        logging.info(f"Mean Recall@{k}: {all_metrics[f'Recall@{k}']:.4f}")
        logging.info(f"MAP@{k}: {all_metrics[f'MAP@{k}']:.4f}")
    
    logging.info("\nPer-Query Recall Summary:")
    for i, row in results_df.iterrows():
        query_summary = f"Query {i+1}: Found {row['Found Count']}/{row['Relevant Count']} relevant assessments"
        for k in k_values:
            query_summary += f", Recall@{k}: {row[f'Recall@{k}']:.2f}"
        logging.info(query_summary)
    
    return all_metrics, results_df

if __name__ == "__main__":
    metrics, results_df = run_enhanced_evaluation([1, 3, 5, 10]) 