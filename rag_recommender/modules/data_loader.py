"""
Data loader for domain keywords and pattern matching data.
Separates data from code logic for better maintainability.
"""
import json
from pathlib import Path
from typing import Dict, List

def load_domain_keywords() -> Dict[str, List[str]]:
    """Load domain-specific keyword mappings from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "domain_keywords.json"
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty dict if file doesn't exist
        return {}

def load_pattern_data() -> Dict:
    """Load pattern matching data from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "pattern_matches.json"
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "direct_match_patterns": {},
            "special_assessments": {},
            "role_skill_mappings": {}
        }

