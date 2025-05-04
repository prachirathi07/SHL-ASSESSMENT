# SHL Assessment Recommendation System: Technical Approach

## Problem Statement
Develop an intelligent recommendation system to help hiring managers find relevant SHL assessments based on natural language queries or job descriptions.

## Data Processing
1. **Data Source**: SHL's product catalog containing 364 assessments
2. **Assessment Representation**: Each assessment includes name, URL, remote testing support, adaptive/IRT support, test type, and duration

## Technical Solution
The solution employs a hybrid recommendation approach combining multiple techniques for optimal performance:

### 1. TF-IDF Vectorization
- Used `scikit-learn`'s TF-IDF Vectorizer to convert assessments and queries into vector representations
- Enhanced with n-gram analysis (1-3) to capture multi-word terms
- Implemented cosine similarity for matching queries to assessments

### 2. Domain-Specific Query Expansion
- Created comprehensive keyword mappings for different domains (programming languages, job roles, skills)
- Dynamically expanded user queries with related industry-specific terminology
- Specialized handling for particular query patterns (banking, sales, tech roles)

### 3. Specialized Pattern Matching
- Implemented regex-based pattern matching for high-precision identification of specific query types
- Direct mapping of assessment recommendations for common query patterns
- Boosting mechanism for assessments highly relevant to detected query patterns

### 4. Hybrid Boosting System
- Duration-based boosting to match assessment length requirements in queries
- Category-specific boosting to prioritize domain-relevant assessments
- Direct word match boosting to increase relevance of exact terminology matches
- Special assessment boosting for high-value assessments in particular domains

### 5. Ensemble Ranking Method
- Combined multiple scoring techniques into a unified ranking algorithm
- Balanced precision of direct matches with recall of semantic matching
- Dynamically adjusts weights based on query characteristics

## Web Application & API
- Flask-based web application with responsive UI for interactive assessment discovery
- RESTful API endpoint for programmatic access to recommendations
- JSON-based response format with comprehensive assessment metadata

## Evaluation Metrics
The system was evaluated using real-world test queries with predefined relevant assessments:

- **Mean Recall@3**: 48.59% (percentage of relevant assessments in top 3 results)
- **Mean Recall@5**: 49.89% (percentage of relevant assessments in top 5 results)
- **Mean Recall@10**: 56.57% (percentage of relevant assessments in top 10 results)
- **Mean Average Precision**: 55.12% (precision averaged over all relevant assessments)

Individual query performance varied, with some test cases (like radio station manager) achieving 100% recall, while others (like banking roles) proved more challenging.

## Technologies Used
- **Languages/Frameworks**: Python, Flask, scikit-learn
- **NLP**: TF-IDF vectorization, regex pattern matching, custom token expansion
- **Web Technologies**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Data Processing**: pandas, numpy
- **Evaluation**: Custom evaluation framework with standard IR metrics

## Extensibility
The system is designed for easy integration of more advanced techniques:
- Embedding-based approaches (SBERT, etc.)
- LLM integration for improved query understanding
- User feedback mechanisms for continuous improvement

## Overview

The SHL Assessment Recommendation System is designed to help hiring managers quickly find relevant assessments based on natural language queries or job descriptions. The system uses a modern Retrieval-Augmented Generation (RAG) approach with Gemini embeddings to provide semantically relevant recommendations.

## Technical Architecture

1. **Data Ingestion Pipeline**
   - Extracted 366 assessments from SHL's product catalog
   - Processed and normalized the data: assessment names, URLs, test types, etc.
   - Created a standardized dataset with all required attributes

2. **Vector Store Creation**
   - Used Google Gemini's embedding model (`models/embedding-001`) to create vector representations
   - Built a FAISS vector index for efficient nearest-neighbor search
   - Combined assessment metadata into indexed records for retrieval

3. **Recommendation Engine**
   - Implemented semantic search capabilities via vector similarity
   - Added post-processing to ensure results match the required format
   - Limited results to at most 10 recommendations per query

4. **API & Interface Layer**
   - Created a Flask web application with a user-friendly interface
   - Implemented RESTful API endpoints following the required specification
   - Added proper error handling and validation

5. **Evaluation System**
   - Implemented Mean Recall@K and MAP@K metrics
   - Created evaluation pipeline with test dataset
   - Documented results and optimization approaches

## Tools & Libraries Used

- **Backend**: Python, Flask
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Gemini API (google-generativeai)
- **Data Processing**: Pandas, NumPy
- **Web Interface**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn, Procfile (cloud platform compatible)

## Performance & Optimization

The system was evaluated using Mean Recall@3 and MAP@3 metrics on a test dataset of queries with known relevant assessments. The initial implementation achieved reasonable performance, which was further improved by:

1. Including assessment URLs in the embedding context
2. Implementing post-processing to enhance result quality
3. Tuning the embedding parameters for better semantic matching

## Production Considerations

- The system is designed for cloud deployment with minimal dependencies
- Environment variables are used for configuration and API keys
- The search index can be easily rebuilt when assessment data is updated
- Documentation is provided for APIs, evaluation, and extensibility

## Future Enhancements

The current implementation could be further improved by:
- Implementing a hybrid search approach (keyword + semantic)
- Adding support for more complex filtering criteria
- Creating a feedback loop for continuous improvement

---

The SHL Assessment Recommendation System demonstrates how modern LLM technologies can be applied to create practical, user-friendly tools that solve real business problems in talent acquisition and assessment.