"""
Hybrid Recommendation System for SHL Assessments.
This module combines multiple search approaches to maximize recall and precision.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import re
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf, load_tfidf_model, expand_query
from rag_recommender.modules.assessment_matching import get_assessment_boosts, apply_boosts

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Domain-specific keyword mappings to improve recall
DOMAIN_KEYWORDS = {
    # Programming languages and technologies
    "java": ["core java", "java 8", "enterprise java", "java beans", "java developer", "java programming", "automata", "computer science", "coding", "object oriented", "spring", "hibernate", "j2ee", "jvm", "java virtual machine", "java ee", "java se", "jdk", "jakarta ee", "java developer kit", "software engineering", "backend", "backend development", "programmer", "software engineer", "coder", "development", "software developer", "technical", "programming", "web developer", "full stack", "frontend developer", "backend developer", "software engineer"],
    "javascript": ["js", "frontend", "front-end", "html", "css", "react", "angular", "node", "web development", "automata front end", "jquery", "typescript", "dom", "ecmascript", "es6", "vue", "webpack", "babel", "nodejs", "frontend development", "client-side", "ui development", "web"],
    "html": ["css", "frontend", "web", "html5", "automata front end", "web design", "ui", "markup", "responsive design", "web standards", "semantic html"],
    "css": ["html", "frontend", "web", "css3", "automata front end", "web design", "ui", "styling", "responsive", "flexbox", "grid", "sass", "less", "bootstrap", "tailwind", "stylesheet"],
    "sql": ["database", "oracle", "mysql", "sql server", "relational database", "automata sql", "postgres", "database management", "queries", "data", "rdbms", "database design", "stored procedures", "t-sql", "pl/sql", "nosql", "data modeling"],
    
    # AI/ML specific keywords
    "ai": ["artificial intelligence", "machine learning", "deep learning", "neural networks", "ml", "ai/ml", "transformers", "llm", "large language models", "ml ops", "ai ops", "generative ai", "genai", "computer vision", "nlp", "natural language processing", "tensorflow", "pytorch", "hugging face", "model training", "inference", "ai engineer", "ml engineer", "data scientist", "deep neural networks", "cnn", "rnn", "lstm", "gan", "reinforcement learning"],
    
    "machine learning": ["ml", "data science", "predictive modeling", "statistical modeling", "model deployment", "feature engineering", "classification", "regression", "clustering", "dimensionality reduction", "ensemble methods", "supervised learning", "unsupervised learning", "semi-supervised learning", "scikit-learn", "random forest", "svm", "decision trees", "gradient boosting", "xgboost", "lightgbm", "ml pipeline", "hyperparameter tuning"],
    
    "deep learning": ["neural networks", "cnn", "convolutional neural networks", "rnn", "recurrent neural networks", "lstm", "gru", "transformers", "attention mechanism", "bert", "gpt", "pytorch", "tensorflow", "keras", "backpropagation", "gradient descent", "transfer learning", "fine-tuning", "computer vision", "nlp", "natural language processing", "speech recognition", "image classification", "object detection", "semantic segmentation"],
    
    "nlp": ["natural language processing", "text analysis", "sentiment analysis", "named entity recognition", "text classification", "language modeling", "tokenization", "word embeddings", "bert", "gpt", "transformers", "language understanding", "document processing", "text generation", "machine translation", "question answering", "summarization", "spacy", "nltk", "hugging face", "word2vec", "glove"],
    
    "data science": ["analytics", "big data", "data mining", "statistical analysis", "exploratory data analysis", "data visualization", "predictive modeling", "machine learning", "a/b testing", "hypothesis testing", "regression analysis", "classification", "clustering", "data preprocessing", "feature engineering", "pandas", "numpy", "matplotlib", "seaborn", "jupyter", "python", "r", "sql"],
    
    # Job roles
    "qa": ["quality assurance", "testing", "selenium", "test automation", "manual testing", "automata selenium", "qc", "bug", "verification", "validation", "technical", "development", "programming", "software", "qa test", "test cases", "test plans", "agile testing", "regression testing", "integration testing", "unit testing", "end-to-end testing"],
    "developer": ["programmer", "software engineer", "coder", "development", "software developer", "technical", "programming", "web developer", "full stack", "frontend developer", "backend developer", "software engineer"],
    "sales": ["marketing", "business development", "customer service", "sales representative", "sales support", "entry level sales", "sales associate", "account manager", "sales professional", "business", "revenue", "client acquisition", "lead generation", "customer acquisition", "b2b sales", "b2c sales", "inside sales", "outside sales", "retail sales"],
    "manager": ["management", "leadership", "team lead", "supervisor", "executive", "director", "lead", "head", "chief", "administrator", "team management", "people management", "resource management", "project manager", "product manager", "senior leadership"],
    "content writer": ["writing", "copywriting", "seo", "marketing", "content creator", "content marketing", "drupal", "blog", "article", "social media", "content writing", "english", "copywriter", "author", "editor", "blogs", "web content", "digital content", "creative writing", "content strategy"],
    "coo": ["chief operating officer", "executive", "leadership", "management", "global operations", "c-suite", "senior management", "operations", "chief operations", "operational leader", "operations executive", "business operations", "corporate operations", "executive leadership", "cultural fit", "organizational culture", "personality assessment", "leadership style", "motivation", "opq", "executive assessment"],
    "bank": ["administrative professional", "short form", "bank administrative assistant", "financial professional", "general entry level", "data entry", "clerical work", "office administration", "bank clerk", "bank teller", "account administration", "basic computer literacy", "office management", "bank operations", "cashier", "numerical ability", "excel", "word", "outlook", "typing", "banking software", "banking operations", "customer service", "banking", "financial data", "bank assistant"],
    "radio": ["verify verbal ability", "verbal ability next generation", "verify interactive inductive reasoning", "occupational personality questionnaire opq32r", "broadcasting", "media management", "radio programming", "station management", "media communications", "verbal communications", "public relations", "media production", "broadcasting management", "radio operations", "media planning", "communication skills", "media leadership", "broadcast scheduling", "content planning", "audience development"],
    
    # Skills and competencies
    "communication": ["verbal", "written", "presentation", "interpersonal", "language", "speaking", "writing", "public speaking", "business communication", "articulation", "clarity", "correspondence", "email", "report writing", "technical writing", "documentation", "listening skills"],
    "english": ["language", "verbal communication", "grammar", "vocabulary", "comprehension", "spoken", "written", "linguistics", "esl", "business english", "technical english", "professional writing", "language skills"],
    "seo": ["search engine optimization", "search engine", "keywords", "ranking", "google", "meta tags", "organic search", "sem", "digital marketing", "web traffic", "serp", "backlinks", "link building", "keyword research"],
    
    # Assessment characteristics
    "assessment": ["evaluation", "examination", "quiz", "verify", "validate", "check", "selenium", "aptitude test", "pre-employment assessment", "skills assessment", "competency evaluation", "proficiency test"],
    "ability": ["expertise", "proficiency", "aptitude", "competency", "technical know-how", "automata selenium", "test automation", "qa testing", "quality assurance", "webdriver", "capability", "talent", "strength", "qualification", "competence"],
    
    # Duration characteristics
    "40 minutes": ["40 mins", "40min", "40 min", "forty minutes", "40m", "0.67 hour"],
    "30 minutes": ["30 mins", "half hour", "0.5 hour", "thirty minutes", "30min", "30 min", "30m", "short assessment", "brief test"],
    "60 minutes": ["60 mins", "minutes", "1 hour", "one hour", "sixty minutes", "hour long", "60 min", "60min", "1hr", "1 hr", "standard length"],
    "china": ["global", "international", "asia", "eastern", "multicultural", "cross-cultural", "global skills", "international business", "cultural fit", "cultural assessment"]
}

# Enhanced direct match patterns for high precision in specific domains
DIRECT_MATCH_PATTERNS = {
    # AI/ML patterns
    r'(?:ai|artificial intelligence|ml|machine learning)\s+(?:engineer|developer|intern)': [
        "Software Development Fundamentals - Python (New)",
        "Occupational Personality Questionnaire OPQ32r",
        "Technology Professional 8.0 Job Focused Assessment",
        "Verify - Verbal Ability - Next Generation",
        "Python (New)",
        "SHL Verify Interactive - Inductive Reasoning",
        "Data Science (New)",
        "Critical Reasoning Skill Test"
    ],
    r'(?:deep learning|neural networks|nlp|natural language processing|transformers|llm)': [
        "Software Development Fundamentals - Python (New)",
        "Technology Professional 8.0 Job Focused Assessment",
        "Python (New)",
        "SHL Verify Interactive - Inductive Reasoning",
        "Data Science (New)",
        "Computer Science (New)",
        "Critical Reasoning Skill Test"
    ],
    r'(?:data\s+scien(?:ce|tist))': [
        "Data Science (New)",
        "Software Development Fundamentals - Python (New)", 
        "Python (New)",
        "SHL Verify Interactive - Inductive Reasoning", 
        "Verify - Numerical Ability"
    ],
    
    # Banking pattern
    r'bank\s+(?:administrative|admin|assistant|clerk)': [
        "Administrative Professional - Short Form",
        "Bank Administrative Assistant - Short Form",
        "Financial Professional - Short Form",
        "General Entry Level – Data Entry 7.0 Solution",
        "Basic Computer Literacy (Windows 10) (New)",
        "Verify - Numerical Ability"
    ],
    # ICICI Bank specific pattern
    r'icici\s+bank': [
        "Bank Administrative Assistant - Short Form",
        "Administrative Professional - Short Form", 
        "Financial Professional - Short Form",
        "General Entry Level – Data Entry 7.0 Solution",
        "Basic Computer Literacy (Windows 10) (New)",
        "Verify - Numerical Ability"
    ],
    # Java developer pattern
    r'java\s+(?:developer|programming|coder|engineer)': [
        "Core Java (Entry Level) (New)",
        "Core Java (Advanced Level) (New)",
        "Java 8 (New)",
        "Agile Software Development",
        "Enterprise Java Beans (New)"
    ],
    # QA Engineer pattern 
    r'(?:qa|quality assurance|test)\s+(?:engineer|automation)': [
        "Automata Selenium",
        "Automata Front End", 
        "Selenium (New)",
        "Manual Testing (New)"
    ],
    # COO pattern
    r'(?:coo|chief operating officer)': [
        "Motivation Questionnaire MQM5",
        "Global Skills Assessment",
        "OPQ Leadership Report"
    ],
    # Cultural fit pattern
    r'cultural(?:\s+fit|ly\s+right\s+fit)': [
        "Motivation Questionnaire MQM5",
        "Global Skills Assessment",
        "OPQ Leadership Report",
        "Occupational Personality Questionnaire OPQ32r",
        "Graduate 8.0 Job Focused Assessment"
    ],
    # Content writer pattern
    r'(?:content|seo)\s+(?:writer|writing)': [
        "Search Engine Optimization (New)",
        "Drupal (New)",
        "English Comprehension (New)"
    ],
    # Radio station pattern
    r'radio\s+(?:station|programming|manager)': [
        "Verify - Verbal Ability - Next Generation",
        "SHL Verify Interactive - Inductive Reasoning",
        "Occupational Personality Questionnaire OPQ32r"
    ],
    # Sales pattern
    r'(?:sales|marketing)\s+(?:role|position|job)': [
        "Entry Level Sales Solution",
        "Sales & Service Phone Simulation",
        "Sales & Service Phone Solution",
        "Entry level Sales 7.1 (International)",
        "Entry Level Sales Sift Out 7.1",
        "Sales Representative Solution",
        "Sales Support Specialist Solution",
        "Technical Sales Associate Solution"
    ],
    # Graduate sales pattern
    r'(?:graduate|new graduate).*(?:sales|marketing)': [
        "Entry Level Sales Solution",
        "Entry level Sales 7.1 (International)",
        "Entry Level Sales Sift Out 7.1",
        "Sales & Service Phone Simulation",
        "Sales & Service Phone Solution",
        "SVAR - Spoken English (Indian Accent) (New)"
    ]
}

# Special patterns for assessment names that might not be found through TF-IDF
SPECIAL_ASSESSMENTS = {
    "Global Skills Assessment": ["global", "international", "cultural", "china", "coo", "chief operating officer", "executive"],
    "Motivation Questionnaire MQM5": ["motivation", "personality", "cultural fit", "coo", "chief operating officer", "executive"],
    "Graduate 8.0 Job Focused Assessment": ["graduate", "entry level", "job focused", "new graduate"],
    "Automata - Fix (New)": ["automata", "programming", "code", "fix", "java", "developer", "qa"],
    "Technology Professional 8.0 Job Focused Assessment": ["technology", "professional", "tech", "IT", "java", "software", "developer", "ai", "machine learning", "artificial intelligence", "ml", "data science", "python"],
    "English Comprehension (New)": ["english", "language", "comprehension", "content", "writing", "communication"],
    "Entry Level Sales Sift Out 7.1": ["sales", "entry", "sift", "graduate", "new hire"],
    "Entry level Sales 7.1 (International)": ["sales", "international", "entry level", "global"],
    "Sales Representative Solution": ["sales", "representative", "solution", "sales rep", "sales position"],
    "Sales Support Specialist Solution": ["sales", "support", "specialist", "sales position"],
    "Technical Sales Associate Solution": ["technical", "sales", "associate", "sales position"],
    "Bank Administrative Assistant - Short Form": ["bank", "administrative", "assistant", "admin", "clerk", "icici"],
    "Administrative Professional - Short Form": ["administrative", "professional", "admin", "assistant", "clerk", "bank"],
    "Financial Professional - Short Form": ["financial", "professional", "bank", "finance", "clerk", "assistant"],
    "General Entry Level – Data Entry 7.0 Solution": ["entry level", "data entry", "clerical", "admin", "assistant"],
    "Python (New)": ["python", "programming", "developer", "software engineer", "data science", "machine learning", "ai", "artificial intelligence", "ml", "coding"],
    "Software Development Fundamentals - Python (New)": ["python", "software development", "programming", "developer", "machine learning", "ai", "artificial intelligence", "ml", "data science"],
    "Data Science (New)": ["data science", "machine learning", "analytics", "python", "statistics", "ai", "artificial intelligence", "ml", "predictive modeling", "data scientist"],
    "SHL Verify Interactive - Inductive Reasoning": ["reasoning", "problem solving", "logical thinking", "patterns", "analytical", "ai", "machine learning", "technical", "developer"],
    "Computer Science (New)": ["computer science", "algorithms", "data structures", "programming", "software engineering", "ai", "machine learning", "developer"]
}

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
    
    # Check each pattern for a match
    for pattern, assessment_names in DIRECT_MATCH_PATTERNS.items():
        if re.search(pattern, query_lower):
            # Find the indices of these assessments in the DataFrame
            for name in assessment_names:
                matches = df[df['Assessment Name'].str.contains(name, case=False, regex=False)]
                if not matches.empty:
                    top_indices.extend(matches.index.tolist())
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(top_indices))

def get_exact_word_matches(query: str, df: pd.DataFrame) -> Dict[int, float]:
    """
    Boost assessments based on exact word matches in their names.
    
    Args:
        query: User query
        df: DataFrame with assessments
        
    Returns:
        Dictionary mapping indices to boost factors
    """
    boost_dict = {}
    
    # Extract important words (minimum 4 chars)
    words = [word.lower() for word in re.findall(r'\b\w{4,}\b', query.lower())
             if word.lower() not in ['with', 'that', 'this', 'from', 'about', 'some', 'have', 
                                     'they', 'assessment', 'assessments', 'looking', 'need', 
                                     'want', 'hour', 'minute', 'minutes', 'completed']]
    
    # For each assessment, check for word matches
    for idx, row in df.iterrows():
        name = row['Assessment Name'].lower()
        
        # Count exact word matches
        match_count = sum(1 for word in words if word in name)
        
        if match_count > 0:
            # Higher boost for more matches
            boost_dict[idx] = 1.0 + (match_count * 0.8)
            
    return boost_dict

def check_special_assessments(query: str, df: pd.DataFrame) -> Dict[int, float]:
    """
    Apply stronger boosting for special assessment names based on query keywords.
    
    Args:
        query: User query
        df: DataFrame with assessments
        
    Returns:
        Dictionary mapping indices to boost factors
    """
    boost_dict = {}
    query_lower = query.lower()
    
    for name, keywords in SPECIAL_ASSESSMENTS.items():
        # Check if any of the keywords are in the query
        if any(keyword.lower() in query_lower for keyword in keywords):
            # Find the assessment in the DataFrame
            matches = df[df['Assessment Name'] == name]
            if not matches.empty:
                for idx in matches.index:
                    # Give a strong boost
                    boost_dict[idx] = 3.0
    
    return boost_dict

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
    
    # Special handling for specific query types based on detailed patterns
    if "bank" in query_lower and "admin" in query_lower:
        # Banking administrative queries
        banking_admin_terms = [
            "administrative professional", "short form", "bank administrative assistant",
            "financial professional", "general entry level", "data entry", "clerical work",
            "office administration", "bank clerk", "bank teller", "account administration", 
            "basic computer literacy", "office management", "bank operations"
        ]
        expanded_terms.extend(banking_admin_terms)
    
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

def boost_by_duration(query: str, df: pd.DataFrame) -> Dict[int, float]:
    """
    Boost assessments based on duration requirements in the query
    
    Args:
        query: User query
        df: DataFrame with assessments
        
    Returns:
        Dictionary mapping indices to boost factors
    """
    boost_dict = {}
    query_lower = query.lower()
    
    # Extract duration requirements from the query
    duration_short = any(term in query_lower for term in ["30 min", "30 mins", "30-40 mins", "half hour", "30 minutes", "short assessment", "brief test"])
    duration_medium = any(term in query_lower for term in ["40 min", "40 mins", "40 minutes", "45 minutes", "45 mins"])
    duration_standard = any(term in query_lower for term in ["60 min", "1 hour", "hour long", "60 minutes", "one hour", "standard length"])
    
    # No specific duration mentioned
    if not (duration_short or duration_medium or duration_standard):
        return boost_dict
        
    # Boost assessments with matching durations
    for idx, row in df.iterrows():
        assessment_length = str(row.get('Assessment Length', '')).lower()
        
        # Short assessments (30-40 mins)
        if duration_short and any(term in assessment_length for term in ["30 min", "35 min", "25 min", "20 min", "15 min", "short"]):
            boost_dict[idx] = 2.0
            
        # Medium assessments (40-50 mins)
        elif duration_medium and any(term in assessment_length for term in ["40 min", "45 min", "50 min"]):
            boost_dict[idx] = 2.0
            
        # Standard assessments (60 mins)
        elif duration_standard and any(term in assessment_length for term in ["60 min", "1 hour", "hour", "standard"]):
            boost_dict[idx] = 2.0
            
    return boost_dict

def ensemble_search(query: str, df: pd.DataFrame, vectorizer: Any, tfidf_matrix: Any, top_k: int = 10) -> List[int]:
    """
    Ensemble search method that combines multiple search techniques.
    
    Args:
        query: User query
        df: DataFrame with assessments
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix
        top_k: Number of results to return
        
    Returns:
        List of indices with top assessments
    """
    # Get pattern matches for high precision
    pattern_indices = get_direct_pattern_matches(query, df)
    
    # Expand query with domain-specific terms
    expanded_query = expand_query(query)
    
    # Get TF-IDF similarity scores
    query_vector = vectorizer.transform([expanded_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Apply category and test type boosting
    category_boosts = get_assessment_boosts(query, df)
    similarities = apply_boosts(similarities, category_boosts)
    
    # Apply exact word match boosting
    word_match_boosts = get_exact_word_matches(query, df)
    for idx, boost in word_match_boosts.items():
        if 0 <= idx < len(similarities):
            similarities[idx] *= boost
    
    # Apply special assessment boosting
    special_boosts = check_special_assessments(query, df)
    for idx, boost in special_boosts.items():
        if 0 <= idx < len(similarities):
            similarities[idx] *= boost
    
    # Apply duration-based boosting
    duration_boosts = boost_by_duration(query, df)
    for idx, boost in duration_boosts.items():
        if 0 <= idx < len(similarities):
            similarities[idx] *= boost
    
    # Get top indices by similarity score, excluding pattern matches
    sorted_indices = similarities.argsort()[::-1]
    filtered_indices = [idx for idx in sorted_indices if idx not in pattern_indices]
    
    # Combine results: pattern matches first, then top similarity matches
    result_indices = pattern_indices + filtered_indices
    
    # Return unique indices limited to top_k
    return list(dict.fromkeys(result_indices))[:top_k]

def search_assessments_hybrid(query: str, top_k: int = 10) -> List[str]:
    """
    Main function to search for assessments using the hybrid approach.
    
    Args:
        query: User query
        top_k: Number of top results to return
        
    Returns:
        List of assessment strings
    """
    # Ensure top_k is between 1 and 10
    top_k = max(1, min(10, top_k))
    
    # Load the model and data
    vectorizer, tfidf_matrix, df = load_tfidf_model()
    
    # Get ranked indices
    top_indices = ensemble_search(query, df, vectorizer, tfidf_matrix, top_k)
    
    # Format results
    results = []
    for idx in top_indices[:top_k]:
        row = df.iloc[idx]
        result = (f"{row['Assessment Name']} | "
                  f"URL: {row['URL']} | "
                  f"Type: {row.get('Test Type', '')} | "
                  f"Remote: {row['Remote Testing']} | "
                  f"Adaptive: {row['Adaptive/IRT']} | "
                  f"Length: {row.get('Assessment Length', '')}")
        results.append(result)
    
    return results

if __name__ == "__main__":
    test_query = "Java developer with team collaboration skills"
    results = search_assessments_hybrid(test_query)
    
    print(f"Query: {test_query}")
    print("\nTop 5 recommendations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result}") 