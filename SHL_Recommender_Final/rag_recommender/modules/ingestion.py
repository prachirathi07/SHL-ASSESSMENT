"""
Data ingestion utilities for SHL Assessment recommendation system.
This module handles loading assessment data from CSV files.
"""
import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def load_assessments(csv_path=None):
    """
    Load assessments from CSV or use default path if not provided.
    
    Args:
        csv_path: Optional path to CSV file with assessment data
        
    Returns:
        DataFrame with assessment data
    """
    if csv_path is None:
        # Default to the assessments.csv in the data directory
        csv_path = Path(__file__).parent.parent / "data" / "assessment.csv"
        
    try:
        logging.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Clean up and standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Fill missing values where appropriate
        for col in df.columns:
            if col not in ['Assessment Name', 'URL']:
                df[col] = df[col].fillna('Not specified')
                
        # Ensure Assessment Name is string
        df['Assessment Name'] = df['Assessment Name'].astype(str)
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading assessment data: {e}")
        raise
        
if __name__ == "__main__":
    # Test the function
    df = load_assessments()
    print(f"Loaded {len(df)} assessments")
    print(df.head()) 