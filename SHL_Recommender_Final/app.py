"""
SHL Assessment Recommendation System
Web application and API endpoints
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
from rag_recommender.modules.url_processor import fetch_job_description, is_valid_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

@app.route('/')
def home():
    """Render the home page with the recommendation form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process form submission and return recommendations."""
    input_method = request.form.get('input_method', 'text')
    query = ''
    
    # Handle different input methods
    if input_method == 'text':
        query = request.form.get('query', '')
        if not query:
            return render_template('index.html', error="Please enter a query or job description.")
    
    elif input_method == 'url':
        url = request.form.get('url', '')
        if not url:
            return render_template('index.html', error="Please enter a URL.")
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            return render_template('index.html', error=f"Error processing URL: {result}")
        
        query = result
    
    try:
        results = search_assessments_hybrid(query, top_k=10)
        formatted_results = []
        
        for result in results:
            # Extract components from the result string
            parts = result.split(' | ')
            name = parts[0]
            
            url = next((p.split(': ')[1] for p in parts if p.startswith('URL: ')), "")
            remote = next((p.split(': ')[1] for p in parts if p.startswith('Remote: ')), "")
            adaptive = next((p.split(': ')[1] for p in parts if p.startswith('Adaptive: ')), "")
            test_type = next((p.split(': ')[1] for p in parts if p.startswith('Type: ')), "")
            duration = next((p.split(': ')[1] for p in parts if p.startswith('Length: ')), "")
            
            formatted_results.append({
                'name': name,
                'url': url,
                'remote': remote,
                'adaptive': adaptive,
                'test_type': test_type,
                'duration': duration
            })
        
        return render_template('results.html', query=query, results=formatted_results)
    
    except Exception as e:
        logging.error(f"Error processing recommendation: {e}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        "status": "healthy",
        "message": "The SHL Assessment Recommendation System API is running."
    })

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for assessment recommendations."""
    data = request.json
    if not data:
        return jsonify({
            "error": "Missing request body",
            "message": "Please provide either a query or URL."
        }), 400
    
    query = None
    
    # Check for URL input
    if 'url' in data:
        url = data['url']
        # Validate and process URL
        if not is_valid_url(url):
            return jsonify({
                "error": "Invalid URL",
                "message": "Please provide a valid HTTP or HTTPS URL."
            }), 400
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            return jsonify({
                "error": "URL processing error",
                "message": result
            }), 400
        
        query = result
    
    # Check for direct query input
    elif 'query' in data:
        query = data['query']
    
    # Neither query nor URL provided
    if not query:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide either 'query' or 'url' in your request."
        }), 400
    
    try:
        results = search_assessments_hybrid(query, top_k=10)
        recommendations = []
        
        for result in results:
            # Extract components from the result string
            parts = result.split(' | ')
            name = parts[0]
            
            url = next((p.split(': ')[1] for p in parts if p.startswith('URL: ')), "")
            remote = next((p.split(': ')[1] for p in parts if p.startswith('Remote: ')), "")
            adaptive = next((p.split(': ')[1] for p in parts if p.startswith('Adaptive: ')), "")
            test_type = next((p.split(': ')[1] for p in parts if p.startswith('Type: ')), "")
            duration = next((p.split(': ')[1] for p in parts if p.startswith('Length: ')), "")
            
            recommendations.append({
                "name": name,
                "url": url,
                "remote_testing": remote == "Yes",
                "adaptive": adaptive == "Yes",
                "test_type": test_type if test_type != "nan" else None,
                "duration": None if duration == "nan" else duration
            })
        
        return jsonify({
            "query": query,
            "recommendations": recommendations
        })
    
    except Exception as e:
        logging.error(f"API Error: {e}")
        return jsonify({
            "error": "Processing error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
