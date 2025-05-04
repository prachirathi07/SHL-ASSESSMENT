"""
Test script for processing non-LinkedIn URLs.
"""
from rag_recommender.modules.url_processor import fetch_job_description
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

def test_non_linkedin_url():
    """
    Process a non-LinkedIn URL and get recommendations
    """
    # Using GitHub Jobs API as an example non-LinkedIn job site
    url = "https://jobs.github.com/positions/43bd7f60-134b-11e7-93e7-c9efb15b4f38"
    
    print(f"\nProcessing non-LinkedIn URL:\n{url}")
    print("=" * 80)
    
    # Try to fetch from URL
    print("Attempting to fetch from URL...")
    success, content = fetch_job_description(url)
    
    if not success:
        print(f"Error: {content}")
        # Fallback to a manually provided job description
        print("\nUsing fallback job description for testing...")
        raw_content = """
        Data Scientist / Machine Learning Engineer
        
        About the Role:
        We are seeking a skilled Data Scientist/Machine Learning Engineer to join our growing team. In this role, you will develop and implement machine learning models and algorithms to extract insights from large datasets.
        
        Responsibilities:
        - Design and develop machine learning models to address business problems
        - Process, cleanse, and verify the integrity of data used for analysis
        - Create data visualizations to communicate findings
        - Work with engineering teams to deploy models to production
        - Stay current with latest ML research and technologies
        
        Requirements:
        - Bachelor's degree in Computer Science, Statistics, or related field
        - 2+ years of experience in data science or machine learning roles
        - Proficiency in Python and data science libraries (NumPy, Pandas, Scikit-learn)
        - Experience with machine learning frameworks like TensorFlow or PyTorch
        - Strong understanding of statistical analysis and modeling techniques
        - Excellent problem-solving and communication skills
        
        Preferred Qualifications:
        - Master's or PhD in a quantitative field
        - Experience with cloud platforms (AWS, GCP, Azure)
        - Knowledge of data engineering and big data tools
        - Background in Natural Language Processing or Computer Vision
        - Experience with data visualization tools
        
        Skills: Python, Machine Learning, Data Science, TensorFlow, Statistical Analysis, Deep Learning, NLP
        """
        success = True
        content = raw_content
    
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
    test_non_linkedin_url() 