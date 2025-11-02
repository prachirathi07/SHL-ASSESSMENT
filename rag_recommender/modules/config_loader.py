"""
Configuration loader for SHL Assessment Recommendation System.
Centralizes all configuration parameters for easy management and tuning.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

_config: Dict[str, Any] = None

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Caches the result for subsequent calls.
    
    Returns:
        Dictionary with configuration values
    """
    global _config
    
    if _config is not None:
        return _config
    
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return _get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
        return _config
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}. Using defaults.")
        return _get_default_config()

def _get_default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "scoring": {
            "skill_match_base": 0.35,
            "skill_match_technical": 0.25,
            "skill_match_personality": 0.3,
            "context_technical": 0.3,
            "context_management": 0.3,
            "context_sales": 0.35,
            "context_administrative": 0.35,
            "context_creative": 0.35,
            "seniority_junior": 0.4,
            "seniority_mid": 0.25,
            "seniority_senior": 0.4,
            "pattern_match_boost": 3.0,
            "duration_match_boost": 0.3,
            "duration_penalty_factor": 0.5,
            "duration_tolerance": 0.3,
        },
        "assessment_matching": {
            "category_match_boost": 2.8,
            "test_type_match_boost": 1.85,
            "duration_close_match": 2.25,
            "duration_approximate_match": 1.75,
            "duration_range_match": 1.4,
            "exact_name_match": 1.5,
            "remote_preference": 1.25,
            "adaptive_preference": 1.25,
            "complete_info_boost": 1.15,
        },
        "model_paths": {
            "tfidf_model": "tfidf_model.pkl",
            "tfidf_matrix": "tfidf_matrix.npy",
            "tfidf_texts": "tfidf_texts.pkl",
            "assessment_df": "assessment_df.pkl",
            "assessment_csv": "rag_recommender/data/assessment.csv",
        },
        "api": {
            "default_top_k": 10,
            "max_top_k": 20,
            "min_top_k": 1,
        },
        "query_expansion": {
            "enabled": True,
            "min_word_length": 4,
        },
        "logging": {
            "level": "INFO",
            "format": "[%(asctime)s]: %(message)s",
        }
    }

def get_config() -> Dict[str, Any]:
    """Get the current configuration (loads if not already loaded)."""
    return load_config()

