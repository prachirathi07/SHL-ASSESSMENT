"""
WSGI entry point for the SHL Assessment Recommendation System.
This file is used for deploying the application to cloud platforms.
"""
from app import app

if __name__ == "__main__":
    app.run() 