"""
Simple script to run the SHL Assessment Recommendation System
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True) 