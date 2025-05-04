"""
Test script for LinkedIn URL processing and recommendations.
"""
import sys
from rag_recommender.modules.url_processor import fetch_job_description
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

def test_linkedin_processing(url):
    """
    Process a LinkedIn URL and get recommendations
    """
    print(f"\nProcessing LinkedIn URL:\n{url}")
    print("=" * 80)
    
    # Fetch job description
    print("Fetching and processing job description...")
    success, content = fetch_job_description(url)
    
    if not success:
        print(f"Error: {content}")
        return
    
    # Display content excerpt
    print("\nExtracted job description (excerpt):")
    print("-" * 80)
    excerpt = content[:500] + "..." if len(content) > 500 else content
    print(excerpt)
    print("-" * 80)
    print(f"Total content length: {len(content)} characters")
    
    # Get recommendations
    print("\nGenerating recommendations...")
    results = search_assessments_hybrid(content, top_k=5)
    
    # Display recommendations
    print("\nTop 5 recommended assessments:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    print("=" * 80)

if __name__ == "__main__":
    # Check if URL was provided as command line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        test_linkedin_processing(url)
    else:
        print("Please provide a LinkedIn job URL as a command line argument.")
        print("Example: python test_linkedin_url.py https://www.linkedin.com/jobs/view/your-job-id") 