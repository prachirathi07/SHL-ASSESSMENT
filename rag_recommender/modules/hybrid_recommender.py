"""
Hybrid Recommendation System for SHL Assessments.
This module combines multiple search approaches to maximize recall and precision.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import re
from sklearn.metrics.pairwise import cosine_similarity

from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf, load_tfidf_model, expand_query
from rag_recommender.modules.assessment_matching import get_assessment_boosts, apply_boosts
from rag_recommender.modules.config_loader import get_config
from rag_recommender.modules.data_loader import load_domain_keywords, load_pattern_data

# Load configuration
CONFIG = get_config()
SCORING = CONFIG.get("scoring", {})

# Load data from external files
DOMAIN_KEYWORDS = load_domain_keywords()
_pattern_data = load_pattern_data()
DIRECT_MATCH_PATTERNS = _pattern_data.get("direct_match_patterns", {})
SPECIAL_ASSESSMENTS = _pattern_data.get("special_assessments", {})
ROLE_SKILL_MAPPINGS = _pattern_data.get("role_skill_mappings", {})

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def get_direct_pattern_matches(query: str, df: pd.DataFrame) -> List[int]:
    """
    Get exact indices of assessments that match specific patterns in the query.
    
    Args:
        query: User query
        df: DataFrame with assessments
        
    Returns:
        List of indices that should be boosted to the top
    """
    query_lower = query.lower()
    top_indices = []
    
    # Check each pattern for a match (compile regex from JSON string)
    for pattern_str, assessment_names in DIRECT_MATCH_PATTERNS.items():
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(query_lower):
                # Find the indices of these assessments in the DataFrame
                for name in assessment_names:
                    matches = df[df['Assessment Name'].str.contains(name, case=False, regex=False)]
                    if not matches.empty:
                        top_indices.extend(matches.index.tolist())
        except re.error:
            # Skip invalid regex patterns
            continue
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(top_indices))

def expand_query(query: str) -> str:
    """
    Expand a query with domain-specific related terms to improve recall
    
    Args:
        query: Original query string
        
    Returns:
        Expanded query with related terms
    """
    expanded_terms = []
    
    # Normalize the query to lowercase for matching
    query_lower = query.lower()
    
    
    # Add domain-specific terms based on keywords in the query
    for keyword, related_terms in DOMAIN_KEYWORDS.items():
        if keyword.lower() in query_lower:
            expanded_terms.extend(related_terms)
    
    # Combine the original query with expanded terms
    expanded_query = query
    if expanded_terms:
        expanded_query = f"{query} {' '.join(expanded_terms)}"
    
    logging.info(f"Expanded query: {query} -> {expanded_query}")
    return expanded_query

def apply_query_context(query: str) -> Dict[str, float]:
    """
    Extract contextual information from the query to better understand user intent.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary of contextual weights for different aspects
    """
    context = {
        "technical": 0.0,
        "management": 0.0,
        "sales": 0.0,
        "administrative": 0.0,
        "creative": 0.0,
        "duration_priority": 0.0,
        "remote_priority": 0.0,
        "experience_level": 0.5,  # Default to mid-level
    }
    
    query_lower = query.lower()
    
    # Technical context signals
    technical_terms = ["developer", "engineer", "programming", "code", "java", "python", 
                      "javascript", "html", "css", "sql", "qa", "testing", "automation"]
    context["technical"] = sum(1 for term in technical_terms if term in query_lower) / len(technical_terms)
    
    # Management context signals
    management_terms = ["manager", "management", "lead", "leadership", "director", 
                       "executive", "coo", "chief", "strategy", "operations"]
    context["management"] = sum(1 for term in management_terms if term in query_lower) / len(management_terms)
    
    # Sales context signals
    sales_terms = ["sales", "marketing", "customer", "service", "client", "account", 
                  "representative", "support", "business development"]
    context["sales"] = sum(1 for term in sales_terms if term in query_lower) / len(sales_terms)
    
    # Administrative context signals
    admin_terms = ["administrative", "admin", "clerical", "assistant", "data entry", 
                  "office", "bank", "clerk", "secretary"]
    context["administrative"] = sum(1 for term in admin_terms if term in query_lower) / len(admin_terms)
    
    # Creative context signals
    creative_terms = ["content", "writer", "writing", "creative", "seo", "english", 
                     "copywriting", "media", "design"]
    context["creative"] = sum(1 for term in creative_terms if term in query_lower) / len(creative_terms)
    
    # Duration priority
    duration_patterns = [r"(\d+)\s*(min|hour|minute)", r"less than (\d+)", r"within (\d+)", 
                         r"(\d+)-(\d+) min", r"quick", r"short", r"brief", r"long"]
    if any(re.search(pattern, query_lower) for pattern in duration_patterns):
        context["duration_priority"] = 0.8
    
    # Experience level detection
    junior_terms = ["entry", "junior", "graduate", "new grad", "trainee", "0-2 year", "beginner"]
    senior_terms = ["senior", "expert", "advanced", "lead", "principal", "experienced", "5+ year"]
    
    if any(term in query_lower for term in junior_terms):
        context["experience_level"] = 0.2
    elif any(term in query_lower for term in senior_terms):
        context["experience_level"] = 0.8
    
    return context

def extract_skill_requirements(query: str) -> Dict[str, float]:
    """
    Extract skill requirements from the query with associated importance weights.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary mapping skills to importance weights
    """
    skill_weights = {}
    query_lower = query.lower()
    
    # Technical skills
    tech_skills = {
        "java": ["java", "j2ee", "spring", "hibernate", "core java", "java 8"],
        "python": ["python", "django", "flask", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
        "javascript": ["javascript", "js", "typescript", "angular", "react", "vue", "node"],
        "web": ["html", "css", "frontend", "web development", "responsive", "ui", "ux"],
        "data": ["sql", "database", "data analysis", "analytics", "reporting", "visualization", "big data"],
        "testing": ["qa", "quality assurance", "testing", "selenium", "automation", "test cases", "manual testing"],
        "devops": ["devops", "ci/cd", "docker", "kubernetes", "jenkins", "git", "aws", "cloud"]
    }
    
    # Functional skills
    func_skills = {
        "sales": ["sales", "account management", "business development", "crm", "negotiation", "closing"],
        "marketing": ["marketing", "digital marketing", "seo", "content", "campaign", "social media"],
        "management": ["management", "leadership", "team lead", "executive", "strategy", "operations"],
        "administrative": ["administrative", "clerical", "data entry", "office", "assistant"],
        "communication": ["communication", "verbal", "written", "presentation", "interpersonal"],
        "customer service": ["customer service", "support", "client", "helpdesk"]
    }
    
    # Detect skills with advanced contextual analysis
    all_skills = {**tech_skills, **func_skills}
    
    for category, terms in all_skills.items():
        # Base detection - direct mentions
        direct_mentions = sum(1 for term in terms if term in query_lower)
        if direct_mentions > 0:
            skill_weights[category] = min(0.9, direct_mentions * 0.3)
        
        # Context-based detection - skill combinations
        if category == "java" and ("backend" in query_lower or "server" in query_lower):
            skill_weights[category] = skill_weights.get(category, 0) + 0.2
        
        if category == "web" and ("frontend" in query_lower or "ui" in query_lower):
            skill_weights[category] = skill_weights.get(category, 0) + 0.2
            
        if category == "testing" and ("quality" in query_lower or "bugs" in query_lower):
            skill_weights[category] = skill_weights.get(category, 0) + 0.2
            
        if category == "management" and ("lead" in query_lower or "team" in query_lower):
            skill_weights[category] = skill_weights.get(category, 0) + 0.3
    
    # Look for emphasis patterns
    emphasis_patterns = [
        (r'strong\s+(\w+)', 0.3),
        (r'expert\s+in\s+(\w+)', 0.4),
        (r'proficient\s+in\s+(\w+)', 0.3),
        (r'experienced\s+(\w+)', 0.3),
        (r'specialized\s+in\s+(\w+)', 0.4),
        (r'knowledge\s+of\s+(\w+)', 0.2)
    ]
    
    for pattern, boost in emphasis_patterns:
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            skill = match.group(1)
            # Find which category this skill belongs to
            for category, terms in all_skills.items():
                if any(skill in term or term in skill for term in terms):
                    skill_weights[category] = min(1.0, skill_weights.get(category, 0) + boost)
    
    # Normalize weights
    if skill_weights:
        max_weight = max(skill_weights.values())
        for skill in skill_weights:
            skill_weights[skill] /= max_weight
    
    return skill_weights

def analyze_role_seniority(query: str) -> Dict[str, float]:
    """
    Analyze the query to determine role seniority and experience level requirements.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with seniority scores
    """
    result = {
        "junior": 0.0,
        "mid": 0.0,
        "senior": 0.0
    }
    
    query_lower = query.lower()
    
    # Junior indicators
    junior_patterns = [
        r'entry\s*level', r'junior', r'graduate', r'trainee', r'intern',
        r'0-1\s*year', r'1\s*year', r'beginner', r'basic', r'fresh'
    ]
    
    # Mid-level indicators
    mid_patterns = [
        r'mid\s*level', r'intermediate', r'experienced', 
        r'2-5\s*years', r'3\s*years', r'4\s*years'
    ]
    
    # Senior indicators
    senior_patterns = [
        r'senior', r'lead', r'expert', r'principal', r'advanced', r'specialist',
        r'5\+\s*years', r'6\+\s*years', r'7\+\s*years', r'manager', r'head', r'chief'
    ]
    
    for pattern in junior_patterns:
        if re.search(pattern, query_lower):
            result["junior"] = max(result["junior"], 0.7)
    
    for pattern in mid_patterns:
        if re.search(pattern, query_lower):
            result["mid"] = max(result["mid"], 0.7)
    
    for pattern in senior_patterns:
        if re.search(pattern, query_lower):
            result["senior"] = max(result["senior"], 0.7)
    
    # Default to mid-level if nothing detected
    if result["junior"] == 0 and result["mid"] == 0 and result["senior"] == 0:
        result["mid"] = 0.5
    
    return result

def hybrid_rank_assessments(query: str, df: pd.DataFrame, vectorizer: Any, tfidf_matrix: Any, top_k: int = 10) -> List[int]:
    """
    Multi-stage assessment ranking system that combines multiple scoring methods.
    
    Args:
        query: User query string
        df: DataFrame with assessments
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix
        top_k: Number of results to return
        
    Returns:
        List of ranked assessment indices
    """
    # Stage 1: Extract query metadata
    context = apply_query_context(query)
    skill_weights = extract_skill_requirements(query)
    seniority = analyze_role_seniority(query)
    
    # Stage 2: TF-IDF base scoring
    expanded_query = expand_query(query)
    query_vector = vectorizer.transform([expanded_query])
    base_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Stage 3: Multi-factor scoring
    final_scores = np.copy(base_scores)
    
    # Pre-process columns for vectorized operations (faster than iterrows)
    df['name_lower'] = df['Assessment Name'].astype(str).str.lower()
    df['test_type_lower'] = df['Test Type'].astype(str).str.lower().fillna('')
    df['duration_numeric'] = pd.to_numeric(df['Assessment Length'], errors='coerce')
    
    # Extract requested duration from query once (outside loop)
    duration_req = None
    if context["duration_priority"] > 0.5:
        for pattern in [r'(\d+)\s*min', r'(\d+)\s*minute', r'within\s*(\d+)']:
            matches = re.findall(pattern, query.lower())
            if matches:
                try:
                    duration_req = int(matches[0])
                    break
                except:
                    continue
    
    # Use itertuples() which is faster than iterrows()
    for row in df.itertuples():
        idx = row.Index
        name = row.name_lower
        test_type = row.test_type_lower
        
        # Initialize combined score components
        skill_match_score = 0.0
        context_match_score = 0.0
        seniority_match_score = 0.0
        
        # Apply skill matching boosts (using config values)
        for skill, weight in skill_weights.items():
            if any(term in name for term in skill.split()):
                skill_match_score += weight * SCORING.get("skill_match_base", 0.35)
            
            # Test type specific matching
            if skill in ['java', 'python', 'javascript'] and "knowledge & skills" in test_type:
                skill_match_score += weight * SCORING.get("skill_match_technical", 0.25)
            
            if skill in ['management', 'leadership'] and "personality" in test_type:
                skill_match_score += weight * SCORING.get("skill_match_personality", 0.3)
        
        # Context-based scoring (using config values)
        for ctx_type, ctx_weight in context.items():
            if ctx_type == "technical" and ctx_weight > 0.3 and "knowledge & skills" in test_type:
                context_match_score += ctx_weight * SCORING.get("context_technical", 0.3)
            
            if ctx_type == "management" and ctx_weight > 0.3 and ("personality" in test_type or "leadership" in name):
                context_match_score += ctx_weight * SCORING.get("context_management", 0.3)
                
            if ctx_type == "sales" and ctx_weight > 0.3 and "sales" in name:
                context_match_score += ctx_weight * SCORING.get("context_sales", 0.35)
                
            if ctx_type == "administrative" and ctx_weight > 0.3 and ("administrative" in name or "data entry" in name):
                context_match_score += ctx_weight * SCORING.get("context_administrative", 0.35)
                
            if ctx_type == "creative" and ctx_weight > 0.3 and ("writing" in name or "english" in name):
                context_match_score += ctx_weight * SCORING.get("context_creative", 0.35)
        
        # Seniority matching (using config values)
        if seniority["junior"] > 0.5 and ("entry level" in name or "basic" in name):
            seniority_match_score += seniority["junior"] * SCORING.get("seniority_junior", 0.4)
        
        if seniority["mid"] > 0.5 and not ("entry" in name or "advanced" in name):
            seniority_match_score += seniority["mid"] * SCORING.get("seniority_mid", 0.25)
            
        if seniority["senior"] > 0.5 and ("advanced" in name or "professional" in name):
            seniority_match_score += seniority["senior"] * SCORING.get("seniority_senior", 0.4)
        
        # Duration matching - from apply_query_context (optimized: duration_req extracted outside loop)
        if context["duration_priority"] > 0.5 and duration_req:
            try:
                # Use pre-processed duration column
                duration = row.duration_numeric
                if pd.notna(duration):
                    duration = float(duration)
                    # Closer to requested duration = higher score (using config values)
                    duration_boost = SCORING.get("duration_match_boost", 0.3)
                    duration_penalty = SCORING.get("duration_penalty_factor", 0.5)
                    duration_tol = SCORING.get("duration_tolerance", 0.3)
                    
                    if duration <= duration_req:
                        final_scores[idx] *= (1 + duration_boost * (1 - (duration_req - duration) / duration_req))
                    else:
                        # Penalty for exceeding
                        final_scores[idx] *= max(duration_tol, 1 - duration_penalty * min(1, (duration - duration_req) / duration_req))
            except (ValueError, TypeError, AttributeError):
                pass
                
        # Combine all scores with TF-IDF base
        combined_boost = 1.0 + (skill_match_score + context_match_score + seniority_match_score)
        final_scores[idx] *= combined_boost
    
    # Stage 4: Apply direct pattern matches (using config value)
    pattern_indices = get_direct_pattern_matches(query, df)
    pattern_boost = SCORING.get("pattern_match_boost", 3.0)
    for idx in pattern_indices:
        if 0 <= idx < len(final_scores):
            final_scores[idx] *= pattern_boost
    
    # Get top indices by final score
    top_indices = final_scores.argsort()[::-1][:top_k]
    
    return top_indices.tolist()

def search_assessments_hybrid(query, top_k=10):
    """
    Enhanced hybrid approach to searching assessments with advanced multi-stage ranking.
    
    Args:
        query: Query string
        top_k: Number of top results to return
        
    Returns:
        List of assessment strings with details
    """
    # Load TF-IDF model and assessment data
    vectorizer, tfidf_matrix, df = load_tfidf_model()
    
    # Get ranked assessment indices
    indices = hybrid_rank_assessments(query, df, vectorizer, tfidf_matrix, top_k=top_k)
    
    # Format and return results
    results = []
    for idx in indices:
        row = df.iloc[idx]
        result = format_assessment_result(row)
        results.append(result)
    
    return results[:top_k]

def format_assessment_result(row):
    """Format a dataframe row as a result string."""
    return (f"{row['Assessment Name']} | "
                  f"URL: {row['URL']} | "
                  f"Type: {row.get('Test Type', '')} | "
                  f"Remote: {row['Remote Testing']} | "
                  f"Adaptive: {row['Adaptive/IRT']} | "
                  f"Length: {row.get('Assessment Length', '')}")

def extract_assessment_name(result):
    """Extract just the assessment name from a result string."""
    if "|" in result:
        return result.split("|")[0].strip()
    return result

if __name__ == "__main__":
    test_query = "Java developer with team collaboration skills"
    results = search_assessments_hybrid(test_query)
    
    print(f"Query: {test_query}")
    print("\nTop 5 recommendations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result}") 