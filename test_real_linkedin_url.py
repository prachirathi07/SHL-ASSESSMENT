"""
Test script for processing the real LinkedIn URL for AI/ML Engineer.
"""
from rag_recommender.modules.url_processor import fetch_job_description, clean_job_description
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

def test_real_linkedin_url():
    """
    Process the real LinkedIn URL for AI/ML Engineer
    """
    url = "https://www.linkedin.com/jobs/view/4175550821/?alternateChannel=search&refId=23Sx6ThMop%2Fh7aNzBrV68g%3D%3D&trackingId=zRVQydddNeDwMsqmTsCnOg%3D%3D"
    
    print(f"\nProcessing Real LinkedIn URL:\n{url}")
    print("=" * 80)
    
    # Try to fetch from URL
    print("Attempting to fetch from URL...")
    success, content = fetch_job_description(url)
    
    # If URL fetch fails or returns minimal content, use the content from the search results
    if not success or len(content) < 100:
        print(f"Could not fetch sufficient content from URL. Using provided content instead.")
        
        # The job description content from the search results
        raw_content = """
        We're looking for AI/ML enthusiasts who build, not just study. If you've implemented transformers from scratch, fine-tuned LLMs, or created innovative ML solutions, we want to see your work!  
          
        Make Sure Before Applying (GitHub Profile Required) 
          
        * Your GitHub must include:
        * At least one substantial ML/DL project with documented results
        * Code demonstrating PyTorch/TensorFlow implementation skills
        * Clear documentation and experiment tracking
        * Bonus: Contributions to ML open-source projects
        * Pin your best projects that showcase:
        * LLM fine-tuning and evaluation
        * Data preprocessing pipelines
        * Model training and optimization
        * Practical applications of AI/ML
        
        Technical Requirements 
          
        * Solid understanding of deep learning fundamentals
        * Python + PyTorch/TensorFlow expertise
        * Experience with Hugging Face transformers
        * Hands-on with large dataset processing
        * NLP/Computer Vision project experience
        
        Education 
          
        * Completed/Pursuing Bachelor's in Computer Science or related field
        * Strong foundation in ML theory and practice
        
        Apply If 
          
        * You have done projects using GenAI, Machine Learning, Deep Learning.
        * You must have strong Python coding experience.
        * Someone who is available immediately to start with us in the office(Hyderabad).
        * Someone who has the hunger to learn something new always and aims to step up at a high pace.
        
        We value quality implementations and thorough documentation over quantity. Show us how you think through problems and implement solutions!  
          
        Skills:- Artificial Intelligence (AI), Machine Learning (ML), Deep Learning, Python, Large Language Models (LLM) tuning, Natural Language Processing (NLP) and CNN
        """
        
        # Clean the content
        content = clean_job_description(raw_content)
    
    # Display content
    print("\nJob Description (excerpt):")
    print("-" * 80)
    excerpt = content[:500] + "..." if len(content) > 500 else content
    print(excerpt)
    print("-" * 80)
    print(f"Total content length: {len(content)} characters")
    
    # Get recommendations
    print("\nGenerating recommendations...")
    results = search_assessments_hybrid(content, top_k=10)
    
    # Display recommendations
    print("\nTop 10 recommended assessments:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    print("=" * 80)

if __name__ == "__main__":
    test_real_linkedin_url() 