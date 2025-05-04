# SHL Assessment Recommendation System

A recommendation system for SHL assessments that helps users find the most appropriate tests based on job descriptions or queries.

## Features

- TF-IDF based recommendation engine with query expansion
- Hybrid recommendation approach with role-based pattern matching
- Web interface for easy querying
- API endpoints for integration with other systems
- URL processing to extract job descriptions from external sources

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface

```
python run.py
```

This will start the Flask web server at http://localhost:5000

### Using the API

Make a POST request to `/api/recommend` with either:

- A direct query: `{"query": "Java developer with Spring experience"}`
- A URL to process: `{"url": "https://example.com/job-description"}`

## How It Works

The system uses a combination of:

1. TF-IDF vector similarity
2. Domain-specific keyword expansion
3. Pattern matching for common roles
4. Role seniority analysis
5. Duration and experience level detection

These approaches are combined to provide highly relevant assessment recommendations.

## Project Structure

- `app.py`: Main Flask application
- `run.py`: Simple script to run the web server
- `rag_recommender/`: Core recommendation modules
  - `modules/`: Individual recommendation components 
  - `data/`: Assessment data files
- `templates/`: HTML templates for the web interface

## Building the TF-IDF Model

If you need to rebuild the TF-IDF model:

```
python -m rag_recommender.build_tfidf
``` 