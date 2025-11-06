"""
SHL Assessment Recommendation System
Web application and API endpoints
"""
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
from rag_recommender.modules.url_processor import fetch_job_description, is_valid_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation System", version="1.0.0")

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates directory
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class RecommendRequest(BaseModel):
    query: Optional[str] = None
    url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with the recommendation form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/recommend', response_class=HTMLResponse)
async def recommend(
    request: Request,
    input_method: str = Form('text'),
    query: Optional[str] = Form(None),
    url: Optional[str] = Form(None)
):
    """Process form submission and return recommendations."""
    processed_query = ''
    
    # Handle different input methods
    if input_method == 'text':
        if not query:
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": "Please enter a query or job description."}
            )
        processed_query = query
    
    elif input_method == 'url':
        if not url:
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": "Please enter a URL."}
            )
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": f"Error processing URL: {result}"}
            )
        
        processed_query = result
    
    try:
        results = search_assessments_hybrid(processed_query, top_k=10)
        formatted_results = []
        
        for result in results:
            # Extract components from the result string
            parts = result.split(' | ')
            name = parts[0]
            
            url_val = next((p.split(': ')[1] for p in parts if p.startswith('URL: ')), "")
            remote = next((p.split(': ')[1] for p in parts if p.startswith('Remote: ')), "")
            adaptive = next((p.split(': ')[1] for p in parts if p.startswith('Adaptive: ')), "")
            test_type = next((p.split(': ')[1] for p in parts if p.startswith('Type: ')), "")
            duration = next((p.split(': ')[1] for p in parts if p.startswith('Length: ')), "")
            
            formatted_results.append({
                'name': name,
                'url': url_val,
                'remote': remote,
                'adaptive': adaptive,
                'test_type': test_type,
                'duration': duration
            })
        
        return templates.TemplateResponse(
            "results.html", 
            {"request": request, "query": processed_query, "results": formatted_results}
        )
    
    except Exception as e:
        logging.error(f"Error processing recommendation: {e}")
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"An error occurred: {str(e)}"}
        )

# API Endpoints
@app.get('/api/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running."""
    return HealthResponse(
        status="healthy",
        message="The SHL Assessment Recommendation System API is running."
    )

@app.post('/api/recommend')
async def api_recommend(request_data: RecommendRequest):
    """API endpoint for assessment recommendations."""
    query = None
    
    # Check for URL input
    if request_data.url:
        url = request_data.url
        # Validate and process URL
        if not is_valid_url(url):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid URL",
                    "message": "Please provide a valid HTTP or HTTPS URL."
                }
            )
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "URL processing error",
                    "message": result
                }
            )
        
        query = result
    
    # Check for direct query input
    elif request_data.query:
        query = request_data.query
    
    # Neither query nor URL provided
    if not query:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required field",
                "message": "Please provide either 'query' or 'url' in your request."
            }
        )
    
    try:
        results = search_assessments_hybrid(query, top_k=10)
        recommendations = []
        
        for result in results:
            # Extract components from the result string
            parts = result.split(' | ')
            name = parts[0]
            
            url_val = next((p.split(': ')[1] for p in parts if p.startswith('URL: ')), "")
            remote = next((p.split(': ')[1] for p in parts if p.startswith('Remote: ')), "")
            adaptive = next((p.split(': ')[1] for p in parts if p.startswith('Adaptive: ')), "")
            test_type = next((p.split(': ')[1] for p in parts if p.startswith('Type: ')), "")
            duration = next((p.split(': ')[1] for p in parts if p.startswith('Length: ')), "")
            
            recommendations.append({
                "name": name,
                "url": url_val,
                "remote_testing": remote == "Yes",
                "adaptive": adaptive == "Yes",
                "test_type": test_type if test_type != "nan" else None,
                "duration": None if duration == "nan" else duration
            })
        
        return {
            "query": query,
            "recommendations": recommendations
        }
    
    except Exception as e:
        logging.error(f"API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing error",
                "message": str(e)
            }
        )
