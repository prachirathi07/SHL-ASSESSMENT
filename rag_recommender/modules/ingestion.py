import pandas as pd
from pathlib import Path
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "assessment.csv"

def generate_url_from_name(name):
    """
    Generate a standardized URL for an assessment based on its name.
    
    Args:
        name: Assessment name
        
    Returns:
        URL string formatted according to SHL's pattern
    """
    # Replace spaces with hyphens, remove special chars, lowercase
    url_path = name.lower()
    url_path = url_path.replace('&', 'and')
    url_path = url_path.replace('.', '-')
    url_path = re.sub(r'[^\w\s-]', '', url_path)  # Remove special chars except hyphens
    url_path = url_path.replace(' ', '-')
    url_path = re.sub(r'-+', '-', url_path)  # Replace multiple hyphens with single hyphen
    
    # Generate the full URL
    return f"https://www.shl.com/solutions/products/product-catalog/view/{url_path}/"

def load_assessments(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the assessments dataset without any preprocessing.
    """
    logging.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Generate URLs if the URL column doesn't exist or has empty values
    if 'URL' not in df.columns:
        df['URL'] = df['Assessment Name'].apply(generate_url_from_name)
    else:
        # Fill missing URL values
        mask = df['URL'].isna() | (df['URL'] == '')
        df.loc[mask, 'URL'] = df.loc[mask, 'Assessment Name'].apply(generate_url_from_name)
    
    logging.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df

def convert_row_to_json(row) -> str:
    """
    Convert each row from the DataFrame to a structured JSON string for embedding.
    Include all required fields: Assessment Name, URL, Remote Testing, Adaptive/IRT, Test Type, Assessment Length
    """
    assessment_json = {
        "data_entity_id": row["data-entity-id"],
        "assessment_name": row["Assessment Name"],
        "url": row["URL"],
        "remote_testing": row["Remote Testing"],
        "adaptive_irt": row["Adaptive/IRT"],
        "duration": row["Assessment Length"],
        "test_type": row["Test Type"]
    }
    return json.dumps(assessment_json)

def preprocess_and_convert_to_json(df: pd.DataFrame):
    """
    Convert all assessments from the DataFrame to JSON format.
    """
    logging.info("Converting each row into JSON format...")
    json_data = df.apply(convert_row_to_json, axis=1).tolist()
    return json_data

# Debug run
if __name__ == "__main__":
    df = load_assessments()
    json_data = preprocess_and_convert_to_json(df)
    print(json_data[:5])  # Print the first 5 JSON entries for debug
