"""
ASGI entry point for the SHL Assessment Recommendation System.
This file is used for deploying the application to cloud platforms.
FastAPI uses ASGI instead of WSGI.
"""
from app import app

# For ASGI servers like uvicorn, gunicorn with uvicorn workers, etc.
# Use: uvicorn wsgi:app or gunicorn wsgi:app -k uvicorn.workers.UvicornWorker 