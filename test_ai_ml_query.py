"""
Test script to check AI/ML job recommendations
"""
from rag_recommender.modules.hybrid_recommender import search_assessments_hybrid

def test_aiml_recommendations():
    """Test recommendations for AI/ML job descriptions"""
    print("\nRecommended Assessments for AI/ML Engineer:")
    print("=" * 80)
    
    job_description = """
    We're looking for AI/ML enthusiasts who build, not just study. If you've implemented transformers from scratch, 
    fine-tuned LLMs, or created innovative ML solutions, we want to see your work!
    
    Make Sure Before Applying (GitHub Profile Required)
    Your GitHub must include:
    At least one substantial ML/DL project with documented results
    Code demonstrating PyTorch/TensorFlow implementation skills
    Clear documentation and experiment tracking
    Bonus: Contributions to ML open-source projects
    
    Pin your best projects that showcase:
    LLM fine-tuning and evaluation
    Data preprocessing pipelines
    Model training and optimization
    Practical applications of AI/ML
    
    Technical Requirements
    Solid understanding of deep learning fundamentals
    Python + PyTorch/TensorFlow expertise
    Experience with Hugging Face transformers
    Hands-on with large dataset processing
    NLP/Computer Vision project experience
    
    Education
    Completed/Pursuing Bachelor's in Computer Science or related field
    Strong foundation in ML theory and practice
    
    Apply If You have done projects using GenAI, Machine Learning, Deep Learning.
    You must have strong Python coding experience.
    
    Skills: Artificial Intelligence (AI), Machine Learning (ML), Deep Learning, Python, 
    Large Language Models (LLM) tuning, Natural Language Processing (NLP) and CNN
    """
    
    results = search_assessments_hybrid(job_description, top_k=10)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
    
    print("=" * 80)

if __name__ == "__main__":
    test_aiml_recommendations() 