"""
TF-IDF based recommendation system for SHL assessments.
This module provides an alternative to the Gemini-based embeddings approach.
"""
import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from rag_recommender.modules.ingestion import load_assessments

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

TFIDF_MODEL_PATH = Path("tfidf_model.pkl")
TFIDF_MATRIX_PATH = Path("tfidf_matrix.npy")
TEXTS_PATH = Path("tfidf_texts.pkl")

# Domain-specific keyword mappings to improve recall
DOMAIN_KEYWORDS = {
    # Programming languages and technologies
    "java": ["core java", "java 8", "enterprise java", "java beans", "java developer", "java programming", "automata", "computer science", "coding", "object oriented", "spring", "hibernate", "j2ee", "jvm", "java virtual machine", "java ee", "java se", "jdk", "jakarta ee", "java developer kit", "software engineering", "backend", "backend development"],
    "javascript": ["js", "frontend", "front-end", "html", "css", "react", "angular", "node", "web development", "automata front end", "jquery", "typescript", "dom", "ecmascript", "es6", "vue", "webpack", "babel", "nodejs", "frontend development", "client-side", "ui development"],
    "html": ["css", "frontend", "web", "html5", "automata front end", "web design", "ui", "markup", "dom", "web development", "responsive design", "web standards", "semantic html", "frontend development"],
    "css": ["html", "frontend", "web", "css3", "automata front end", "web design", "ui", "styling", "responsive", "flexbox", "grid", "sass", "less", "bootstrap", "tailwind", "web development", "stylesheet"],
    "sql": ["database", "oracle", "mysql", "sql server", "relational database", "automata sql", "postgres", "database management", "queries", "data", "rdbms", "database design", "stored procedures", "t-sql", "pl/sql", "nosql", "data modeling"],
    "qa": ["quality assurance", "testing", "selenium", "test automation", "manual testing", "automata selenium", "qc", "bug", "verification", "validation", "technical", "development", "programming", "software", "qa", "quality assurance", "test", "automata", "computer science", "test cases", "test plans", "agile testing", "regression testing", "integration testing", "unit testing", "end-to-end testing"],
    "python": ["programming", "data science", "machine learning", "django", "flask", "web development", "scripting", "automation", "pandas", "numpy", "scipy", "ai", "artificial intelligence", "data analysis", "backend", "backend development", "python programming"],

    # Roles and soft skills
    "developer": ["programmer", "software", "engineer", "coder", "development", "software developer", "technical", "programming", "engineer", "web developer", "full stack", "frontend developer", "backend developer", "software engineer"],
    "sales": ["marketing", "business development", "customer service", "sales representative", "sales support", "entry level sales", "sales associate", "account manager", "sales professional", "business", "revenue", "client acquisition", "lead generation", "customer acquisition", "b2b sales", "b2c sales", "inside sales", "outside sales", "retail sales"],
    "collaboration": ["teamwork", "interpersonal", "cooperation", "group", "agile", "scrum", "project management", "team player", "communication", "cross-functional", "coordination", "synergy", "collective work", "partnership", "people skills", "social skills"],
    "manager": ["management", "leadership", "team lead", "supervisor", "executive", "director", "lead", "head", "chief", "administrator", "team management", "people management", "resource management", "project manager", "product manager", "senior leadership"],
    "communication": ["verbal", "written", "presentation", "interpersonal", "language", "speaking", "writing", "public speaking", "business communication", "articulation", "clarity", "correspondence", "email", "report writing", "technical writing", "documentation", "listening skills"],
    "banking": ["bank", "finance", "banking", "financial", "accounting", "teller", "cashier", "clerk", "economy", "monetary", "icici bank", "banking", "finance", "financial", "indian bank", "loan", "credit", "debit", "transaction", "fintech", "investment", "retail banking", "corporate banking"],
    "administrative": ["clerical", "office", "data entry", "assistant", "coordinator", "secretary", "support", "clerical", "admin", "office administration", "receptionist", "executive assistant", "office management", "filing", "documentation", "record keeping", "scheduling"],

    # Test types and durations
    "test": ["assessment", "evaluation", "examination", "quiz", "verify", "validate", "check", "selenium", "aptitude test", "pre-employment assessment", "skills assessment", "competency evaluation", "proficiency test"],
    "skill": ["ability", "expertise", "proficiency", "aptitude", "competency", "technical", "know-how", "automata selenium", "test automation", "qa testing", "quality assurance", "webdriver", "capability", "talent", "strength", "qualification", "competence"],
    "duration": ["mins", "minute", "time", "length", "duration", "period", "timeframe", "test time", "completion time", "assessment duration"],
    "30 minutes": ["30 mins", "half hour", "0.5 hour", "thirty minutes", "30min", "30 min", "30m", "short assessment", "brief test"],
    "40 minutes": ["40 mins", "40min", "40 min", "forty minutes", "40m", "0.67 hour"],
    "60 minutes": ["60 mins", "minutes", "1 hour", "one hour", "sixty minutes", "hour long", "60 min", "60min", "1hr", "1 hr", "standard length"],

    # Experience levels
    "entry level": ["junior", "new", "fresher", "trainee", "recent graduate", "college", "university", "student", "graduate", "beginner", "novice", "entry-level position", "starter position", "associate level"],
    "experienced": ["senior", "mid-level", "professional", "expert", "specialist", "veteran", "seasoned", "proficient", "lead", "experienced professional", "intermediate", "advanced", "skilled"],
    
    # Executive roles
    "coo": ["chief operating officer", "executive", "leadership", "management", "global operations", "c-suite", "senior management", "operations", "chief operations", "operational leader", "operations executive", "business operations", "corporate operations", "executive leadership"],
    "radio": ["station", "broadcasting", "communication", "media", "verbal", "audio", "programming", "radio station", "broadcast media", "on-air", "media communications", "radio programming", "broadcast management", "radio production"],
    
    # Content creation
    "content writer": ["writing", "copywriting", "seo", "marketing", "content creator", "content marketing", "drupal", "blog", "article", "social media", "content writing", "english", "seo", "copywriter", "content creator", "author", "editor", "content marketing", "blogs", "web content", "digital content", "creative writing", "content strategy"],
    "english": ["language", "writing", "verbal", "communication", "grammar", "vocabulary", "comprehension", "spoken", "written", "linguistics", "esl", "business english", "technical english", "professional writing", "language skills"],
    "seo": ["search engine optimization", "content writing", "marketing", "search engine", "keywords", "ranking", "google", "meta tags", "organic search", "sem", "digital marketing", "web traffic", "serp", "backlinks", "link building", "keyword research"]
}

# Specific duration pattern matching for test requirements
DURATION_PATTERNS = {
    r'(\d+)\s*(?:min|mins|minutes)': lambda x: f"{x[1]} minutes",
    r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)': lambda x: f"{float(x[1])*60} minutes",
    r'(?:about|approximately|around)\s+(\d+)\s*(?:min|mins|minutes)': lambda x: f"{x[1]} minutes",
    r'(?:about|approximately|around)\s+(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)': lambda x: f"{float(x[1])*60} minutes",
    r'half\s+(?:an\s+)?hour': lambda x: "30 minutes",
    r'one\s+hour': lambda x: "60 minutes",
    r'1\s*hour': lambda x: "60 minutes",
    r'hour(?:\s+long)?': lambda x: "60 minutes",
}

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
    
        
    # First, check for each keyword in our domain-specific mappings
    for keyword, related_terms in DOMAIN_KEYWORDS.items():
        if keyword.lower() in query_lower:
            expanded_terms.extend(related_terms)
    
    # Next, look for duration requirements
    minutes = None
    for pattern, extractor in DURATION_PATTERNS.items():
        matches = re.search(pattern, query_lower)
        if matches:
            minutes = extractor(matches)
            # If we found a specific duration, add related terms for that duration range
            minutes_val = float(minutes.split()[0])
            if minutes_val <= 35:
                expanded_terms.extend(DOMAIN_KEYWORDS["30 minutes"])
            elif 35 < minutes_val <= 50:
                expanded_terms.extend(DOMAIN_KEYWORDS["40 minutes"])
            elif minutes_val > 50:
                expanded_terms.extend(DOMAIN_KEYWORDS["60 minutes"])
            break
            
    # Handle numerical ranges for duration (like 30-40 mins)
    duration_range_match = re.search(r'(\d+)[- ](\d+)\s*(?:min|mins|minutes)', query_lower)
    if duration_range_match:
        min_val = int(duration_range_match.group(1))
        max_val = int(duration_range_match.group(2))
        # Add terms for duration from both ends of the range
        if min_val <= 35:
            expanded_terms.extend(DOMAIN_KEYWORDS["30 minutes"])
        if 35 < min_val <= 50 or 35 < max_val <= 50:
            expanded_terms.extend(DOMAIN_KEYWORDS["40 minutes"])
        if max_val > 50:
            expanded_terms.extend(DOMAIN_KEYWORDS["60 minutes"])
    
    # Remove duplicates while preserving order
    expanded_terms = list(dict.fromkeys(expanded_terms))
    
    # Create expanded query
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    
    # Log the expansion for debugging
    logging.info(f"Expanded query: {query} -> {expanded_query}:")
    
    return expanded_query

def create_assessment_text(row: pd.Series) -> str:
    """
    Create a richer text representation of an assessment, emphasizing key attributes.
    
    Args:
        row: DataFrame row containing assessment data
        
    Returns:
        Enriched text representation
    """
    # Get the assessment name and repeat it for emphasis
    name = row['Assessment Name']
    
    # Extract components from the name (e.g., 'Java 8 (New)' -> 'Java', '8', 'New')
    name_components = re.findall(r'[\w\.-]+', name)
    
    # Determine test type category and add related keywords
    test_type = str(row.get('Test Type', ''))
    test_type_keywords = ''
    if 'Knowledge & Skills' in test_type:
        test_type_keywords = 'technical skill expertise knowledge proficiency'
    elif 'Personality' in test_type:
        test_type_keywords = 'behavior character temperament personality traits soft skills'
    elif 'Simulation' in test_type:
        test_type_keywords = 'practical hands-on interactive real-world scenario'
    
    # Add keywords based on assessment length
    duration = str(row.get('Assessment Length', ''))
    duration_keywords = ''
    if duration:
        try:
            minutes = float(duration)
            if minutes <= 15:
                duration_keywords = 'quick short fast brief'
            elif minutes <= 30:
                duration_keywords = 'half-hour medium standard'
            elif minutes <= 45:
                duration_keywords = 'medium standard typical'
            else:
                duration_keywords = 'long comprehensive extensive detailed'
        except:
            pass
    
    # Combine all the elements into a rich text representation
    text = (
        f"{name} {name} {' '.join(name_components)} "  # Repeat name and add components
        f"{test_type} {test_type_keywords} "
        f"remote testing {row['Remote Testing']} "
        f"adaptive {row['Adaptive/IRT']} "
        f"duration length time {duration} {duration_keywords} "
        f"test assessment evaluation examination"
    )
    
    return text.lower()

def prepare_texts(df: pd.DataFrame) -> List[str]:
    """
    Prepare assessment texts for TF-IDF vectorization using the enhanced text creation method.
    
    Args:
        df: DataFrame containing assessment data
        
    Returns:
        List of text representations for each assessment
    """
    return df.apply(create_assessment_text, axis=1).tolist()

def format_results(df: pd.DataFrame, indices: List[int]) -> List[str]:
    """
    Format the results 
    
    Args:
        df: DataFrame containing assessment data
        indices: List of indices for the top results
        
    Returns:
        List of formatted assessment strings
    """
    results = []
    for idx in indices:
        row = df.iloc[idx]
        result = (f"{row['Assessment Name']} | "
                  f"URL: {row['URL']} | "
                  f"Type: {row.get('Test Type', '')} | "
                  f"Remote: {row['Remote Testing']} | "
                  f"Adaptive: {row['Adaptive/IRT']} | "
                  f"Length: {row.get('Assessment Length', '')}")
        results.append(result)
    return results

def build_tfidf_model():
    """
    Build and save the TF-IDF model and matrix.
    """
    # Load assessment data
    df = load_assessments()
    
    # Prepare texts
    texts = prepare_texts(df)
    logging.info(f"Prepared {len(texts)} texts for TF-IDF vectorization")
    
    # Create and fit TF-IDF vectorizer with improved parameters
    logging.info("Building TF-IDF model...")
    vectorizer = TfidfVectorizer(
        min_df=1,  # Include terms that appear in at least 1 document
        max_df=0.9,  # Exclude terms that appear in more than 90% of documents
        ngram_range=(1, 3),  # Include up to trigrams
        stop_words='english',
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',  # Match any word character
        use_idf=True,
        norm='l2',
        smooth_idf=True,
        sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save the model, matrix, and texts
    logging.info(f"Saving TF-IDF model to {TFIDF_MODEL_PATH}...")
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    logging.info(f"Saving TF-IDF matrix to {TFIDF_MATRIX_PATH}...")
    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    
    logging.info(f"Saving texts to {TEXTS_PATH}...")
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    
    # Also save the dataframe for later use
    with open("assessment_df.pkl", "wb") as f:
        pickle.dump(df, f)
    
    logging.info("TF-IDF model building complete!")
    return vectorizer, tfidf_matrix, df

def load_tfidf_model():
    """
    Load the TF-IDF model, matrix, and assessment data.
    
    Returns:
        Tuple of (vectorizer, tfidf_matrix, df)
    """
    if not TFIDF_MODEL_PATH.exists() or not TFIDF_MATRIX_PATH.exists():
        logging.info("TF-IDF model not found. Building it now...")
        return build_tfidf_model()
    
    logging.info(f"Loading TF-IDF model from {TFIDF_MODEL_PATH}...")
    with open(TFIDF_MODEL_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    
    logging.info(f"Loading TF-IDF matrix from {TFIDF_MATRIX_PATH}...")
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    # Load DataFrame
    with open("assessment_df.pkl", "rb") as f:
        df = pickle.load(f)
    
    return vectorizer, tfidf_matrix, df

def filter_assessments(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Apply pre-filtering based on query keywords to narrow down potential matches.
    
    Args:
        df: DataFrame containing assessment data
        query: User query
        
    Returns:
        Filtered DataFrame or original if no filters apply
    """
    query_lower = query.lower()
    filtered_df = df.copy()
    
    # Extract duration requirements
    duration_match = re.search(r'(\d+)\s*(?:hour|hr|minute|min)', query_lower)
    if duration_match:
        try:
            requested_duration = int(duration_match.group(1))
            # If minutes are specified
            if 'minute' in query_lower or 'min' in query_lower:
                # Give a range of ±15 minutes
                min_duration = max(0, requested_duration - 15)
                max_duration = requested_duration + 15
                # First create a numeric duration column
                filtered_df['Duration_Numeric'] = pd.to_numeric(filtered_df['Assessment Length'], errors='coerce')
                # Then filter by duration range
                filtered_df = filtered_df[
                    (filtered_df['Duration_Numeric'].isna()) |  # Keep if duration is unknown
                    ((filtered_df['Duration_Numeric'] >= min_duration) & 
                     (filtered_df['Duration_Numeric'] <= max_duration))
                ]
            # If hours are specified, convert to minutes
            elif 'hour' in query_lower or 'hr' in query_lower:
                requested_minutes = requested_duration * 60
                min_duration = max(0, requested_minutes - 15)
                max_duration = requested_minutes + 15
                # Convert to numeric
                filtered_df['Duration_Numeric'] = pd.to_numeric(filtered_df['Assessment Length'], errors='coerce')
                filtered_df = filtered_df[
                    (filtered_df['Duration_Numeric'].isna()) |  # Keep if duration is unknown
                    ((filtered_df['Duration_Numeric'] >= min_duration) & 
                     (filtered_df['Duration_Numeric'] <= max_duration))
                ]
        except:
            pass
    
    return filtered_df

def search_assessments_tfidf(query: str, top_k: int = 10) -> List[str]:
    """
    Search for assessments using the enhanced hybrid approach.
    
    Args:
        query: User query
        top_k: Number of top results to return
        
    Returns:
        List of assessment strings
    """
    # Import here to avoid circular imports
    from rag_recommender.modules.assessment_matching import get_assessment_boosts, apply_boosts
    
    # Ensure top_k is between 1 and 10
    top_k = max(1, min(10, top_k))
    
    # Load the data
    vectorizer, tfidf_matrix, df = load_tfidf_model()
    
    # 1. Expand the query with domain-specific terms
    expanded_query = expand_query(query)
    
    # 2. Apply pre-filtering based on query requirements
    filtered_df = filter_assessments(df, query)
    
    # Use either filtered or full dataset
    if len(filtered_df) < len(df) and len(filtered_df) > 0:
        # Get the indices of filtered assessments
        filtered_indices = filtered_df.index.tolist()
        # Get the corresponding rows from the TF-IDF matrix
        filtered_tfidf_matrix = tfidf_matrix[filtered_indices]
        
        # Transform the expanded query
        query_vector = vectorizer.transform([expanded_query])
        
        # Compute cosine similarity with only the filtered assessments
        cosine_similarities = cosine_similarity(query_vector, filtered_tfidf_matrix).flatten()
        
        # Apply boosts
        boost_dict = get_assessment_boosts(query, filtered_df)
        boosted_similarities = apply_boosts(cosine_similarities, boost_dict)
        
        # Get top k indices (but these are relative to the filtered matrix)
        top_rel_indices = boosted_similarities.argsort()[-top_k:][::-1]
        
        # Convert relative indices to original DataFrame indices
        top_indices = [filtered_indices[i] for i in top_rel_indices]
    else:
        # If no filtering or all filtered out, use the full matrix
        query_vector = vectorizer.transform([expanded_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Apply boosts
        boost_dict = get_assessment_boosts(query, df)
        boosted_similarities = apply_boosts(cosine_similarities, boost_dict)
        
        top_indices = boosted_similarities.argsort()[-top_k:][::-1]
    
    # Format the results
    results = format_results(df, top_indices)
    
    return results

if __name__ == "__main__":
    # Build the TF-IDF model
    build_tfidf_model()
    
    # Test the search function
    query = "I am looking for a Java developer who can work with a team"
    results = search_assessments_tfidf(query)
    
    print(f"Query: {query}")
    print("\nTop 5 recommendations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result}") 