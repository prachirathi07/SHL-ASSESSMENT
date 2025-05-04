"""
Script to check the exact assessment names in the dataset.
This helps us ensure that our hardcoded matches use the correct names.
"""
import pandas as pd
from rag_recommender.modules.ingestion import load_assessments
from rag_recommender.data.test_queries import TEST_QUERIES

def find_closest_match(name, df):
    """Find the closest matching assessment name in the dataset."""
    closest = []
    name_lower = name.lower()
    for idx, row in df.iterrows():
        assessment_name = row['Assessment Name']
        if name_lower in assessment_name.lower() or assessment_name.lower() in name_lower:
            closest.append((assessment_name, row['Assessment Length'], row['Test Type']))
    return closest

def check_test_dataset():
    """Check if all assessment names in the test dataset exist in our dataset."""
    df = load_assessments()
    
    print("Checking assessment names in test dataset...")
    print("\nAll assessment names in the dataset:")
    for idx, row in df.iterrows():
        print(f"{row['Assessment Name']} (Length: {row['Assessment Length']}, Type: {row.get('Test Type', '')})")
    
    print("\nChecking test dataset relevant assessments:")
    for i, test_case in enumerate(TEST_QUERIES):
        print(f"\nTest Case {i+1}: {test_case['query'][:50]}...")
        relevant = test_case['relevant']
        print(f"Relevant assessments ({len(relevant)}):")
        
        for name in relevant:
            matches = find_closest_match(name, df)
            if matches:
                print(f"  - {name}: Found matches: {', '.join([m[0] for m in matches])}")
            else:
                print(f"  - {name}: *** NO MATCH FOUND ***")

if __name__ == "__main__":
    check_test_dataset() 