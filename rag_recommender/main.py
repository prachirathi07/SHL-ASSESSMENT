from modules.rag_pipeline import search_assessments

def main():
    query = "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins. "
    results = search_assessments(query)
    
    print("\nTop Recommendations:\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")

if __name__ == "__main__":
    main()
