"""
SHL Assessment Recommendation System
FastAPI Web application and API endpoints
"""
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid
from rag_recommender.modules.url_processor import fetch_job_description, is_valid_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Hybrid recommendation system for SHL assessments",
    version="1.0.0"
)

# Configure CORS - Allow localhost for dev and Vercel domains for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "*"  # For production, replace with specific domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class RecommendRequest(BaseModel):
    query: Optional[str] = Field(None, description="Text query for job description")
    url: Optional[str] = Field(None, description="URL to fetch job description from")

class HealthResponse(BaseModel):
    status: str
    message: str

class RecommendationItem(BaseModel):
    name: str
    url: str
    remote_testing: bool
    adaptive: bool
    test_type: Optional[str] = None
    duration: Optional[str] = None

class RecommendResponse(BaseModel):
    query: str
    recommendations: list[RecommendationItem]

# Helper function to format results
def format_recommendations(results):
    """Format recommendation results into structured list."""
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
        
        recommendations.append(RecommendationItem(
            name=name,
            url=url,
            remote_testing=remote == "Yes",
            adaptive=adaptive == "Yes",
            test_type=test_type if test_type != "nan" else None,
            duration=None if duration == "nan" else duration
        ))
    
    return recommendations

@app.get("/", response_class=HTMLResponse)
async def home():
    """Render the home page with the recommendation form."""
    try:
        # Use Path for better cross-platform compatibility
        template_path = Path("templates/index.html")
        if template_path.exists():
            # Read with explicit UTF-8 encoding
            content = template_path.read_text(encoding="utf-8")
            return HTMLResponse(content=content)
        else:
            # If template not found, redirect to docs
            return RedirectResponse(url="/docs")
    except Exception as e:
        logging.error(f"Error loading template: {e}")
        # Fallback to simple HTML or redirect to docs
        return HTMLResponse(
            content="<h1>SHL Assessment Recommender API</h1><p><a href='/docs'>View API Documentation</a></p><p><a href='/api/health'>Health Check</a></p>",
            status_code=200
        )

@app.post("/recommend", response_class=HTMLResponse)
async def recommend_form(
    input_method: str = Form("text"),
    query: Optional[str] = Form(None),
    url: Optional[str] = Form(None)
):
    """Process form submission and return recommendations (HTML response)."""
    actual_query = ''
    
    # Handle different input methods
    if input_method == 'text':
        if not query:
            return HTMLResponse(
                content="<html><body><h1>Error</h1><p>Please enter a query or job description.</p></body></html>",
                status_code=400
            )
        actual_query = query
    
    elif input_method == 'url':
        if not url:
            return HTMLResponse(
                content="<html><body><h1>Error</h1><p>Please enter a URL.</p></body></html>",
                status_code=400
            )
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            return HTMLResponse(
                content=f"<html><body><h1>Error</h1><p>Error processing URL: {result}</p></body></html>",
                status_code=400
            )
        
        actual_query = result
    
    try:
        results = search_assessments_hybrid(actual_query, top_k=10)
        formatted_results = format_recommendations(results)
        
        # Simple HTML response (you can enhance this)
        html_content = f"""
        <html>
        <head><title>Recommendations</title></head>
        <body>
            <h1>Recommendations for: {actual_query[:50]}...</h1>
            <ul>
        """
        for rec in formatted_results:
            html_content += f"<li><strong>{rec.name}</strong> - {rec.url}</li>"
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        logging.error(f"Error processing recommendation: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>An error occurred: {str(e)}</p></body></html>",
            status_code=500
        )

# API Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running."""
    return HealthResponse(
        status="healthy",
        message="The SHL Assessment Recommendation System API is running."
    )

@app.post("/api/recommend", response_model=RecommendResponse)
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
                detail="Invalid URL. Please provide a valid HTTP or HTTPS URL."
            )
        
        # Fetch job description from URL
        success, result = fetch_job_description(url)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"URL processing error: {result}"
            )
        
        query = result
    
    # Check for direct query input
    elif request_data.query:
        query = request_data.query
    
    # Neither query nor URL provided
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'query' or 'url' in your request."
        )
    
    try:
        results = search_assessments_hybrid(query, top_k=10)
        recommendations = format_recommendations(results)
        
        return RecommendResponse(
            query=query,
            recommendations=recommendations
        )
    
    except Exception as e:
        logging.error(f"API Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, reload=True)
