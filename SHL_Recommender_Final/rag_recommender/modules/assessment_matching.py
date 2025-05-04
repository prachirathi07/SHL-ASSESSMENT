"""
Specialized Assessment Matching Module for SHL Recommendation System.
This module contains custom rules for matching assessments to specific queries.
"""
import re
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Domain-specific boosted keywords for different assessment categories
ASSESSMENT_CATEGORIES = {
    "Java Programming": [
        "Java 8", "Core Java", "Enterprise Java", "Java Beans", 
        "Automata - Fix", "Computer Science", "Java", "Software Development", 
        "Agile Software Development", "Programming", "Object Oriented", "Spring",
        "J2EE", "Hibernate", "Technology Professional", "Backend Development",
        "JVM", "Java Virtual Machine", "Java EE", "Java SE", "JDK", "Jakarta EE"
    ],
    "JavaScript Development": [
        "JavaScript", "JS", "Frontend", "Front-end", "HTML", "CSS", 
        "Angular", "React", "Vue", "Node.js", "TypeScript", "Web Development",
        "DOM", "ECMAScript", "ES6", "Webpack", "Babel", "Client-side",
        "Browser", "UI Development", "Frontend Development", "jQuery"
    ],
    "Web Development": [
        "HTML", "CSS", "HTML5", "CSS3", "Frontend", "Frontend Development",
        "Web Design", "UI", "Responsive Design", "Web Standards", "Semantic HTML",
        "Bootstrap", "Tailwind", "SASS", "LESS", "Flexbox", "Grid"
    ],
    "QA Engineer": [
        "Quality Assurance", "Testing", "Selenium", "Test Automation", "Manual Testing", 
        "Automata Selenium", "QC", "Bug", "Verification", "Validation", "QA", 
        "Test Cases", "Test Plans", "Regression Testing", "Integration Testing",
        "Unit Testing", "End-to-End Testing", "WebDriver", "Automated Testing",
        "Test Management", "Bug Tracking", "JIRA", "TestRail", "Cucumber", "BDD", "TDD"
    ],
    "Database": [
        "SQL", "Database", "Oracle", "MySQL", "SQL Server", "PostgreSQL", 
        "Relational Database", "RDBMS", "Automata SQL", "Data", "Database Design", 
        "Stored Procedures", "T-SQL", "PL/SQL", "NoSQL", "Data Modeling",
        "MongoDB", "Cassandra", "Redis", "Database Administration", "Query Optimization"
    ],
    "Content Creation": [
        "Writing", "Copywriting", "SEO", "Content Writer", "Content Creation",
        "Content Marketing", "Drupal", "Blog", "Article", "Social Media",
        "Content Strategy", "Content Writing", "English", "Web Content", 
        "Digital Content", "Creative Writing", "Editing", "Proofreading",
        "Content Management", "CMS", "WordPress", "Search Engine Optimization"
    ],
    "Sales": [
        "Sales", "Entry Level Sales", "Marketing", "Business Development",
        "Customer Service", "Account Management", "Sales Representative", 
        "Sales Support", "Business", "Revenue", "Client Acquisition", 
        "Lead Generation", "Customer Acquisition", "B2B Sales", "B2C Sales",
        "Inside Sales", "Outside Sales", "Retail Sales", "Sales Associate",
        "Account Manager", "Sales Professional", "Sales Executive"
    ],
    "Banking": [
        "Banking", "Bank", "Finance", "Financial", "Accounting", "Teller",
        "Cashier", "Clerk", "Economy", "Monetary", "ICICI Bank", "Indian Bank",
        "Loan", "Credit", "Debit", "Transaction", "FinTech", "Investment",
        "Retail Banking", "Corporate Banking", "Financial Analysis", "Banking Operations",
        "Risk Management", "Financial Regulations", "Banking Technology"
    ],
    "Administrative": [
        "Administrative", "Clerical", "Office", "Data Entry", "Assistant",
        "Coordinator", "Secretary", "Support", "Admin", "Office Administration",
        "Receptionist", "Executive Assistant", "Office Management", "Filing",
        "Documentation", "Record Keeping", "Scheduling", "Administrative Support",
        "Front Office", "Back Office", "General Administrative", "Clerical Work"
    ],
    "Executive": [
        "COO", "Chief Operating Officer", "Executive", "Leadership", "Management",
        "Operations", "C-suite", "Senior Management", "Global Operations",
        "Chief Operations", "Operational Leader", "Operations Executive",
        "Business Operations", "Corporate Operations", "Executive Leadership",
        "CEO", "CFO", "CTO", "Senior Executive", "Director", "VP", "Vice President"
    ],
    "Media": [
        "Radio", "Station", "Broadcasting", "Communication", "Media",
        "Verbal", "Audio", "Programming", "Radio Station", "Broadcast Media",
        "On-air", "Media Communications", "Radio Programming", "Broadcast Management",
        "Radio Production", "Media Production", "Content Production", "Media Management"
    ]
}

# Test types to match in queries
TEST_TYPES = {
    "Knowledge & Skills": [
        "knowledge", "skills", "technical", "proficiency", "expertise",
        "abilities", "competency", "aptitude", "capability", "qualification",
        "technical assessment", "skills test", "knowledge test", "proficiency exam"
    ],
    "Personality & Behavior": [
        "personality", "behavior", "character", "temperament", "psychological",
        "traits", "motivation", "cultural fit", "OPQ", "psychometric",
        "behavioral assessment", "personality test", "behavioral profile"
    ],
    "Simulations": [
        "simulation", "interactive", "hands-on", "scenario", "practical",
        "role play", "real-world", "situational", "case study", "simulation-based",
        "virtual job tryout", "work sample", "job simulation"
    ],
    "Biodata & Situational Judgement": [
        "situational", "judgment", "judgement", "biodata", "biographical",
        "decision making", "situational judgment test", "SJT", "scenario-based",
        "work behavior", "problem-solving", "decision assessment"
    ]
}

# Regex pattern to extract duration from query
DURATION_PATTERN = re.compile(r'(\d+)\s*(?:min|mins|minutes|hour|hours|hr|hrs)')

def get_assessment_boosts(query: str, results_df) -> Dict[int, float]:
    """
    Generate boosting factors for assessments based on query match.
    
    Args:
        query: User query
        results_df: Dataframe with assessment results
        
    Returns:
        Dictionary of boost factors by index
    """
    boost_dict = {}
    query_lower = query.lower()
    
    # Check for category matches and boost accordingly
    for idx, row in results_df.iterrows():
        boost_factor = 1.0
        assessment_name = row['Assessment Name']
        test_type = str(row['Test Type']).lower()
        remote = str(row['Remote Testing']).lower()
        adaptive = str(row['Adaptive/IRT']).lower()
        duration = row['Assessment Length']
        
        # 1. Check for category matches
        for category, keywords in ASSESSMENT_CATEGORIES.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                # If the assessment name contains any word from this matched category
                if any(word.lower() in assessment_name.lower() for word in keywords):
                    boost_factor *= 2.8  # More aggressive boost for direct category match
                
        # 2. Check for test type matches
        if test_type != "nan":
            for type_name, type_keywords in TEST_TYPES.items():
                if type_name.lower() == test_type:
                    if any(keyword.lower() in query_lower for keyword in type_keywords):
                        boost_factor *= 1.85  # Boost if test type matches query requirements
        
        # 3. Check for duration requirements with enhanced pattern matching
        duration_match = DURATION_PATTERN.search(query_lower)
        if duration_match and str(duration) != "nan":
            try:
                # Convert query duration to minutes
                query_duration = int(duration_match.group(1))
                if "hour" in duration_match.group(0) or "hr" in duration_match.group(0):
                    query_duration *= 60  # Convert hours to minutes
                
                assessment_duration = float(duration)
                
                # Apply more granular duration matching with flexible tolerance
                duration_diff = abs(assessment_duration - query_duration)
                
                # More flexible scoring based on how close the duration is
                if duration_diff <= 5:  # Very close match (within 5 minutes)
                    boost_factor *= 2.25
                elif duration_diff <= 15:  # Close match (within 15 minutes)
                    boost_factor *= 1.75
                elif duration_diff <= 30:  # Approximate match (within 30 minutes)
                    boost_factor *= 1.4
            except (ValueError, TypeError):
                pass  # Skip if we can't convert the duration to a number
        
        # 4. Boost exact assessment name matches
        for word in query_lower.split():
            if len(word) > 3 and word.lower() in assessment_name.lower():
                boost_factor *= 1.5  # Boost for direct name match
        
        # 5. Remote testing preference - if specified in query
        if "remote" in query_lower and remote == "yes":
            boost_factor *= 1.25
        
        # 6. Adaptive testing preference - if specified in query
        if "adaptive" in query_lower and adaptive == "yes":
            boost_factor *= 1.25
            
        # Add small boost for assessments with complete information (better quality)
        if test_type != "nan" and str(duration) != "nan" and remote != "nan":
            boost_factor *= 1.15
            
        # Store the boost factor if it's different from the default
        if boost_factor != 1.0:
            boost_dict[idx] = boost_factor
    
    return boost_dict

def apply_boosts(similarities: List[float], boost_dict: Dict[int, float]) -> List[float]:
    """
    Apply boosting factors to similarity scores.
    
    Args:
        similarities: Original similarity scores
        boost_dict: Dictionary of boosting factors by index
        
    Returns:
        Boosted similarity scores
    """
    boosted = similarities.copy()
    for idx, boost in boost_dict.items():
        if 0 <= idx < len(boosted):
            boosted[idx] = boosted[idx] * boost
    return boosted 