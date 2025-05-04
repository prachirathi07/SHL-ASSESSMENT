import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K: The fraction of relevant assessments retrieved in the top K recommendations.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant (ground truth) assessment names
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score between 0.0 and 1.0
    """
    if not relevant:
        return 0.0
    
    # Extract assessment names only from recommended items
    if "|" in recommended[0]:
        recommended_names = [item.split("|")[0].strip() for item in recommended[:k]]
    else:
        recommended_names = recommended[:k]
    
    # Count matches between recommended and relevant items
    matches = sum(1 for item in relevant if any(item.lower() in rec.lower() for rec in recommended_names))
    return matches / len(relevant)

def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K: The fraction of recommendations that are relevant.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant (ground truth) assessment names
        k: Number of top recommendations to consider
    
    Returns:
        Precision@K score between 0.0 and 1.0
    """
    if k == 0 or not recommended or not relevant:
        return 0.0
    
    # Extract assessment names only from recommended items
    if "|" in recommended[0]:
        recommended_names = [item.split("|")[0].strip() for item in recommended[:k]]
    else:
        recommended_names = recommended[:k]
    
    # Count matches between recommended and relevant items
    matches = sum(1 for item in relevant if any(item.lower() in rec.lower() for rec in recommended_names))
    return matches / min(k, len(recommended))

def average_precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Average Precision@K: The average of precision values at positions where relevant items are found.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant (ground truth) assessment names
        k: Number of top recommendations to consider
    
    Returns:
        Average Precision@K score between 0.0 and 1.0
    """
    if not relevant or not recommended:
        return 0.0
    
    # Extract assessment names only from recommended items
    if "|" in recommended[0]:
        recommended_names = [item.split("|")[0].strip() for item in recommended[:k]]
    else:
        recommended_names = recommended[:k]
    
    score = 0.0
    num_hits = 0
    
    for i, rec in enumerate(recommended_names):
        # Check if current recommendation is relevant
        is_relevant = any(r.lower() in rec.lower() for r in relevant)
        if is_relevant:
            num_hits += 1
            # Precision at current position
            precision_at_i = num_hits / (i + 1)
            score += precision_at_i
    
    # Normalize by the number of relevant items (capped at k)
    return score / min(len(relevant), k) if num_hits > 0 else 0.0

def mean_metrics(results: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
    """
    Calculate Mean Recall@K and MAP@K across a set of queries.
    
    Args:
        results: List of dictionaries with 'recommended' and 'relevant' lists
        k: Number of top recommendations to consider
    
    Returns:
        Dictionary with 'mean_recall_at_k' and 'map_at_k' values
    """
    recall_scores = []
    ap_scores = []
    
    for result in results:
        recommended = result['recommended']
        relevant = result['relevant']
        
        recall = recall_at_k(recommended, relevant, k)
        ap = average_precision_at_k(recommended, relevant, k)
        
        recall_scores.append(recall)
        ap_scores.append(ap)
    
    return {
        'mean_recall_at_k': np.mean(recall_scores) if recall_scores else 0.0,
        'map_at_k': np.mean(ap_scores) if ap_scores else 0.0
    }

def evaluate_test_set(search_func, test_queries: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
    """
    Evaluate the recommendation system on a test set.
    
    Args:
        search_func: Function that takes a query and returns recommendations
        test_queries: List of dictionaries with 'query' and 'relevant' fields
        k: Number of top recommendations to consider
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = []
    
    for test_case in test_queries:
        query = test_case['query']
        relevant = test_case['relevant']
        
        try:
            recommended = search_func(query)
            results.append({
                'query': query,
                'recommended': recommended,
                'relevant': relevant
            })
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
    
    metrics = mean_metrics(results, k)
    logging.info(f"Evaluation Results: Mean Recall@{k}={metrics['mean_recall_at_k']:.4f}, MAP@{k}={metrics['map_at_k']:.4f}")
    
    return metrics
