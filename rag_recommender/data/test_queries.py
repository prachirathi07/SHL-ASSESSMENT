"""
Test dataset for evaluating the SHL Assessment recommendation system.
Each test case has a query and a list of relevant assessments.
"""

TEST_QUERIES = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "relevant": [
            "Automata - Fix (New)",
            "Core Java (Entry Level) (New)",
            "Java 8 (New)",
            "Core Java (Advanced Level) (New)",
            "Agile Software Development",
            "Technology Professional 8.0 Job Focused Assessment",
            "Computer Science (New)"
        ]
    },
    {
        "query": "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
        "relevant": [
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
        ]
    },
    {
        "query": "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour",
        "relevant": [
            "Motivation Questionnaire MQM5",
            "Global Skills Assessment",
            "Graduate 8.0 Job Focused Assessment"
        ]
    },
    {
        "query": "Content Writer required, expert in English and SEO.",
        "relevant": [
            "Drupal (New)",
            "Search Engine Optimization (New)",
            "Administrative Professional - Short Form",
            "Entry Level Sales Sift Out 7.1",
            "General Entry Level – Data Entry 7.0 Solution"
        ]
    },
    {
        "query": "Find me 1 hour long assessment for a QA Engineer with experience in JavaScript, CSS, HTML, Selenium WebDriver and SQL server",
        "relevant": [
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
        ]
    },
    {
        "query": "ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long",
        "relevant": [
            "Administrative Professional - Short Form",
            "Verify - Numerical Ability",
            "Financial Professional - Short Form",
            "Bank Administrative Assistant - Short Form",
            "General Entry Level – Data Entry 7.0 Solution",
            "Basic Computer Literacy (Windows 10) (New)"
        ]
    },
    {
        "query": "Looking for a radio station programming manager with excellent communication skills, ability to work with sales teams, and people management experience",
        "relevant": [
            "Verify - Verbal Ability - Next Generation",
            "SHL Verify Interactive - Inductive Reasoning",
            "Occupational Personality Questionnaire OPQ32r"
        ]
    }
] 