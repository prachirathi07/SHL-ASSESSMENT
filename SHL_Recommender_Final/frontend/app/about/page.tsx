export default function About() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12">
        <h1 className="text-4xl font-extrabold text-text-primary sm:text-5xl mb-4 text-center relative inline-block mx-auto">
          <span className="relative z-10">About SHL Assessment Recommender</span>
          <div className="absolute -bottom-2 left-0 w-full h-1 bg-gradient-to-r from-primary to-secondary rounded-full"></div>
        </h1>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="card glassmorphism mb-8">
          <h2 className="text-2xl font-bold mb-4 text-primary">What is the SHL Assessment Recommender?</h2>
          <p className="mb-4 text-text-secondary">
            The SHL Assessment Recommender is an advanced tool designed to help HR professionals, 
            recruiters, and hiring managers find the most appropriate SHL assessments for their 
            job roles or candidates.
          </p>
          <p className="text-text-secondary">
            Using natural language processing and recommendation algorithms, our system analyzes 
            job descriptions or queries to suggest the most relevant SHL assessments that will 
            help you identify the best candidates for your positions.
          </p>
        </div>

        <div className="card glassmorphism mb-8">
          <h2 className="text-2xl font-bold mb-4 text-primary">How It Works</h2>
          <p className="mb-4 text-text-secondary">
            Our recommendation system uses a hybrid approach combining multiple techniques:
          </p>
          <ol className="list-decimal pl-6 space-y-3 mb-4 text-text-secondary">
            <li>
              <span className="font-semibold text-primary">TF-IDF Vector Similarity</span>: We use text analysis to understand 
              the semantic meaning of your job descriptions and match them with appropriate assessments.
            </li>
            <li>
              <span className="font-semibold text-primary">Domain-Specific Expansion</span>: We expand queries with industry-specific 
              terminology to improve recall and relevance.
            </li>
            <li>
              <span className="font-semibold text-primary">Pattern Matching</span>: For common roles, we apply specialized patterns 
              to ensure the most appropriate assessments are recommended.
            </li>
            <li>
              <span className="font-semibold text-primary">Role Analysis</span>: We detect seniority levels, duration requirements, 
              and specific skill needs in your queries.
            </li>
          </ol>
          <p className="text-text-secondary">
            These approaches are combined to provide you with highly relevant assessment recommendations 
            tailored to your specific requirements.
          </p>
        </div>

        <div className="card glassmorphism mb-8">
          <h2 className="text-2xl font-bold mb-4 text-primary">How to Use</h2>
          <p className="mb-4 text-text-secondary">
            Using the SHL Assessment Recommender is simple:
          </p>
          <ol className="list-decimal pl-6 space-y-3 text-text-secondary">
            <li>
              <span className="font-semibold text-primary">Enter a Query</span>: Type a job description, role requirements, 
              or specific skills you're looking to assess.
            </li>
            <li>
              <span className="font-semibold text-primary">Or Provide a URL</span>: Alternatively, paste a URL to a job 
              posting, and our system will extract the relevant information.
            </li>
            <li>
              <span className="font-semibold text-primary">Review Recommendations</span>: The system will analyze your input 
              and suggest the most relevant SHL assessments for your needs.
            </li>
            <li>
              <span className="font-semibold text-primary">Access Assessments</span>: Click on any recommended assessment 
              to learn more about it and how to incorporate it into your hiring process.
            </li>
          </ol>
        </div>
      </div>
    </div>
  )
} 