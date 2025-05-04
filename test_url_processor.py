"""
Test script for the URL processor module
"""
import argparse
from rag_recommender.modules.url_processor import fetch_job_description
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

def test_url_to_recommendations(url):
    """
    Extract job description from URL and get recommendations
    """
    print(f"Fetching job description from URL: {url}")
    success, content = fetch_job_description(url)
    
    if not success:
        print(f"Error: {content}")
        return
    
    print("\nExtracted job description (excerpt):")
    print("=" * 80)
    print(content[:500] + "..." if len(content) > 500 else content)
    print("=" * 80)
    print(f"\nTotal length: {len(content)} characters")
    
    print("\nGenerating recommendations...")
    results = search_assessments_hybrid(content, top_k=5)
    
    print("\nTop 5 recommended assessments:")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test URL processing and recommendations")
    parser.add_argument("url", help="URL of a job description to process")
    args = parser.parse_args()
    
    test_url_to_recommendations(args.url) 