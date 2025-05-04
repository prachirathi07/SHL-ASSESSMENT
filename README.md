# SHL Assessment Recommendation System

This system recommends relevant SHL assessments based on natural language queries, job descriptions, or job description URLs. It uses an advanced hybrid recommendation approach combining TF-IDF, domain-specific pattern matching, and specialized boosting mechanisms to provide highly accurate assessment recommendations.

## Overview

Hiring managers often struggle to find the right assessments for roles they're hiring for. This intelligent recommendation system simplifies the process by:

1. Taking natural language queries, job descriptions, or job description URLs as input
2. Analyzing key requirements and keywords
3. Recommending relevant SHL assessments with all required attributes

## Features

- **Advanced natural language processing** - Understand hiring needs from free-text queries
- **URL processing** - Extract job descriptions directly from URLs
- **Domain-specific knowledge** - Built-in understanding of job roles, skills, and assessment characteristics
- **Hybrid recommendation approach** - Combines multiple techniques for optimal results
- **Duration-aware matching** - Considers time requirements in recommendations
- **Web UI** - Clean, responsive interface for easy interaction
- **API access** - RESTful endpoints for programmatic integration

## Performance

Evaluated on standard test queries, the system achieves:

- Mean Recall@3: 48.59%
- Mean Recall@5: 49.89%
- Mean Recall@10: 56.57%
- Mean Average Precision: 55.12%

Individual query performance varies by domain, with technical roles like QA Engineers and specialized positions like Radio Station Managers achieving the highest recall scores.

## Architecture

- **Core Recommendation Engine**: Hybrid approach combining TF-IDF vectorization, pattern matching, and boosting
- **Web Application**: Flask-based interface with responsive design
- **API Layer**: RESTful endpoints for programmatic access
- **URL Processing Module**: Extracts job descriptions from web pages
- **Evaluation Module**: Tools for performance measurement and continuous improvement

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. Clone this repository
```bash
git clone https://github.com/yourusername/shl-recommendation-system.git
cd shl-recommendation-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the web application
```bash
python run.py
```

4. Access the application at http://localhost:5000

## Usage

### Web Interface
- Enter a natural language query describing the role, skills needed, or assessment requirements, or
- Provide a URL to a job description page
- View the recommended assessments with all relevant details
- Click on assessment URLs to explore detailed information

### API
The system provides a RESTful API endpoint for programmatic access:

#### Using a text query:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "Looking for a Java developer with experience in Spring"}' http://localhost:5000/api/recommend
```

#### Using a URL:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/job-posting/java-developer"}' http://localhost:5000/api/recommend
```

Response format:
```json
{
  "query": "Looking for a Java developer with experience in Spring",
  "recommendations": [
    {
      "name": "Core Java (Entry Level) (New)",
      "url": "https://www.shl.com/...",
      "remote_testing": true,
      "adaptive": false,
      "test_type": "Knowledge & Skills",
      "duration": "13.0"
    },
    ...
  ]
}
```

## Evaluation

The system was evaluated using industry-standard metrics on a test set with ground truth relevant assessments:

- **Recall@K**: Measures the percentage of relevant assessments found within the top K recommendations
- **Mean Average Precision**: Measures the precision averaged over all relevant assessments
- **Per-query performance**: Analyzes performance across different query domains

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **scikit-learn**: For TF-IDF vectorization and similarity calculations
- **BeautifulSoup**: For parsing job descriptions from URLs
- **pandas/numpy**: For data processing
- **Bootstrap 5**: For responsive UI design
- **RESTful API**: For programmatic access

## Future Improvements

- Integration with more sophisticated NLP models
- User feedback mechanisms for continuous improvement
- Support for more languages and regional assessment variations
- Expanded domain coverage for specialized roles
- Enhanced URL processing capabilities for various job description formats
