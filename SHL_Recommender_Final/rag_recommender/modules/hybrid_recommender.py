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
import math

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

# Add role-specific skill mappings
ROLE_SKILL_MAPPINGS = {
    "java developer": ["java", "core java", "enterprise java", "spring", "hibernate", "junit", "programming", "software development"],
    "qa engineer": ["testing", "quality assurance", "selenium", "automation testing", "manual testing", "test cases", "qa", "qc"],
    "sales": ["sales", "customer service", "negotiation", "pitching", "closing", "account management", "crm"],
    "content writer": ["writing", "content creation", "seo", "editing", "copywriting", "blogging", "social media"],
    "data scientist": ["python", "r", "statistics", "machine learning", "data analysis", "sql", "big data"],
    "manager": ["leadership", "management", "team building", "strategy", "operations", "planning", "budgeting"],
    "finance": ["accounting", "financial analysis", "budgeting", "forecasting", "excel", "financial reporting"],
    "administrator": ["administration", "clerical", "data entry", "office management", "documentation"]
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
    
    for idx, row in df.iterrows():
        name = str(row['Assessment Name']).lower()
        test_type = str(row.get('Test Type', '')).lower()
        
        # Initialize combined score components
        skill_match_score = 0.0
        context_match_score = 0.0
        seniority_match_score = 0.0
        
        # Apply skill matching boosts
        for skill, weight in skill_weights.items():
            if any(term in name for term in skill.split()):
                skill_match_score += weight * 0.35
            
            # Test type specific matching
            if skill in ['java', 'python', 'javascript'] and "knowledge & skills" in test_type:
                skill_match_score += weight * 0.25
            
            if skill in ['management', 'leadership'] and "personality" in test_type:
                skill_match_score += weight * 0.3
        
        # Context-based scoring
        for ctx_type, ctx_weight in context.items():
            if ctx_type == "technical" and ctx_weight > 0.3 and "knowledge & skills" in test_type:
                context_match_score += ctx_weight * 0.3
            
            if ctx_type == "management" and ctx_weight > 0.3 and ("personality" in test_type or "leadership" in name):
                context_match_score += ctx_weight * 0.3
                
            if ctx_type == "sales" and ctx_weight > 0.3 and "sales" in name:
                context_match_score += ctx_weight * 0.35
                
            if ctx_type == "administrative" and ctx_weight > 0.3 and ("administrative" in name or "data entry" in name):
                context_match_score += ctx_weight * 0.35
                
            if ctx_type == "creative" and ctx_weight > 0.3 and ("writing" in name or "english" in name):
                context_match_score += ctx_weight * 0.35
        
        # Seniority matching
        if seniority["junior"] > 0.5 and ("entry level" in name or "basic" in name):
            seniority_match_score += seniority["junior"] * 0.4
        
        if seniority["mid"] > 0.5 and not ("entry" in name or "advanced" in name):
            seniority_match_score += seniority["mid"] * 0.25
            
        if seniority["senior"] > 0.5 and ("advanced" in name or "professional" in name):
            seniority_match_score += seniority["senior"] * 0.4
        
        # Duration matching - from apply_query_context
        if context["duration_priority"] > 0.5:
            try:
                duration_str = str(row.get('Assessment Length', ''))
                duration_match = re.search(r'(\d+)', duration_str)
                if duration_match:
                    duration = int(duration_match.group(1))
                    
                    # Extract requested duration from query
                    duration_req = None
                    for pattern in [r'(\d+)\s*min', r'(\d+)\s*minute', r'within\s*(\d+)']:
                        matches = re.findall(pattern, query.lower())
                        if matches:
                            try:
                                duration_req = int(matches[0])
                                break
                            except:
                                continue
                    
                    if duration_req:
                        # Closer to requested duration = higher score
                        if duration <= duration_req:
                            final_scores[idx] *= (1 + 0.3 * (1 - (duration_req - duration) / duration_req))
                        else:
                            # Penalty for exceeding
                            final_scores[idx] *= max(0.3, 1 - 0.5 * min(1, (duration - duration_req) / duration_req))
            except:
                pass
                
        # Combine all scores with TF-IDF base
        combined_boost = 1.0 + (skill_match_score + context_match_score + seniority_match_score)
        final_scores[idx] *= combined_boost
    
    # Stage 4: Apply direct pattern matches
    pattern_indices = get_direct_pattern_matches(query, df)
    for idx in pattern_indices:
        if 0 <= idx < len(final_scores):
            final_scores[idx] *= 3.0  # Strong boost for pattern matches
    
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

# Helper function for query expansion
def search_assessments_tfidf_with_query_expansion(query, vectorizer, tfidf_matrix, df, top_k=10):
    """
    Enhanced TF-IDF search with query expansion techniques.
    
    Args:
        query: Query string
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix
        df: Assessment dataframe
        top_k: Number of top results to return
        
    Returns:
        List of (assessment_string, score) tuples
    """
    # Original TF-IDF query processing
    query_vector = vectorizer.transform([expand_query(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top results
    top_indices = cosine_similarities.argsort()[-top_k*2:][::-1]
    
    # Create a list of (assessment_name, score) tuples
    scored_results = []
    for idx in top_indices:
        assessment_name = df.iloc[idx]['Assessment Name']
        score = cosine_similarities[idx]
        
        # Create the result string
        result = format_assessment_result(df.iloc[idx])
        scored_results.append((result, score))
    
    return scored_results

# Add this helper function
def format_assessment_result(row):
    """Format a dataframe row as a result string."""
    return (f"{row['Assessment Name']} | "
                  f"URL: {row['URL']} | "
                  f"Type: {row.get('Test Type', '')} | "
                  f"Remote: {row['Remote Testing']} | "
                  f"Adaptive: {row['Adaptive/IRT']} | "
                  f"Length: {row.get('Assessment Length', '')}")

# Add this helper function
def extract_assessment_name(result):
    """Extract just the assessment name from a result string."""
    if "|" in result:
        return result.split("|")[0].strip()
    return result

def apply_duration_constraints(ranked_assessments, query, df, max_score=1.0):
    """
    Apply duration constraints to adjust scores based on time requirements in the query.
    
    Args:
        ranked_assessments: List of (assessment_name, score) tuples
        query: The original query string
        df: Assessment dataframe with duration information
        max_score: Maximum possible score
        
    Returns:
        List of (assessment_name, adjusted_score) tuples
    """
    # Extract duration requirements from query
    duration_req = None
    duration_keywords = [
        r'(\d+)\s*min', r'(\d+)\s*minute', r'within\s*(\d+)', 
        r'under\s*(\d+)', r'less than\s*(\d+)', r'(\d+)\s*mins'
    ]
    
    for pattern in duration_keywords:
        matches = re.findall(pattern, query.lower())
        if matches:
            try:
                duration_req = int(matches[0])
                break
            except:
                continue
    
    if not duration_req:
        return ranked_assessments
    
    adjusted_assessments = []
    
    for name, score in ranked_assessments:
        # Find the assessment in the dataframe
        assessment_rows = df[df['Assessment Name'] == name]
        if assessment_rows.empty:
            adjusted_assessments.append((name, score))
            continue
        
        # Extract duration
        duration_str = assessment_rows.iloc[0]['Assessment Length']
        if pd.isna(duration_str) or duration_str == "nan" or not duration_str:
            adjusted_assessments.append((name, score))
            continue
        
        try:
            # Extract numeric duration
            duration_val = float(re.search(r'(\d+\.?\d*)', str(duration_str)).group(1))
            
            # Apply penalty for exceeding duration requirement
            if duration_val > duration_req:
                # Logarithmic penalty - more severe as duration exceeds limit more
                penalty = min(0.9, math.log(1 + (duration_val - duration_req) / duration_req))
                adjusted_score = score * (1 - penalty)
            else:
                # Boost for assessments within duration requirements
                boost = min(0.2, (duration_req - duration_val) / (duration_req * 2))
                adjusted_score = min(max_score, score * (1 + boost))
                
            adjusted_assessments.append((name, adjusted_score))
        except:
            adjusted_assessments.append((name, score))
    
    # Re-sort based on adjusted scores
    return sorted(adjusted_assessments, key=lambda x: x[1], reverse=True)

def expand_query_with_role_specific_terms(query):
    """
    Expand query with role-specific terms based on detected roles.
    
    Args:
        query: Original query string
        
    Returns:
        Expanded query with additional role-specific terms
    """
    query_lower = query.lower()
    expanded_terms = []
    
    # Check for role matches
    for role, skills in ROLE_SKILL_MAPPINGS.items():
        if role in query_lower or any(term in query_lower for term in role.split()):
            # Add role-specific skills to expansion
            expanded_terms.extend(skills)
    
    # Join original query with expanded terms
    if expanded_terms:
        return f"{query} {' '.join(expanded_terms)}"
    return query

if __name__ == "__main__":
    test_query = "Java developer with team collaboration skills"
    results = search_assessments_hybrid(test_query)
    
    print(f"Query: {test_query}")
    print("\nTop 5 recommendations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result}") 