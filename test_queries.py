"""
Simple test script to demonstrate the enhanced recommender system.
"""
from rag_recommender.modules.tfidf_recommender import search_assessments_tfidf

def test_queries():
    """Test a few different queries to show the system's capabilities."""
    queries = [
        "Java developer with good team collaboration skills",
        "Content Writer with experience in English and SEO optimization",
        "QA Engineer with knowledge of Selenium and JavaScript",
        "Sales professional for entry level position",
        "COO who would be a good cultural fit for our company"
    ]
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        
        results = search_assessments_tfidf(query, top_k=5)
        
        print("\nTop 5 recommendations:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")

if __name__ == "__main__":
    test_queries() 