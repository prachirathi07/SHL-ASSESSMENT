'use client'

import { useState, FormEvent } from 'react'
import axios from 'axios'

interface Assessment {
  name: string
  url: string
  remote_testing?: boolean
  remote?: string
  adaptive?: string
  test_type?: string
  duration?: string
}

export default function Home() {
  const [inputMethod, setInputMethod] = useState<'text' | 'url'>('text')
  const [query, setQuery] = useState('')
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<Assessment[] | null>(null)
  const [error, setError] = useState('')

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResults(null)

    try {
      let response;
      console.log(`Submitting ${inputMethod === 'text' ? 'query' : 'url'}: ${inputMethod === 'text' ? query : url}`);
      
      if (inputMethod === 'text') {
        response = await axios.post('/api/recommend', { query })
      } else {
        response = await axios.post('/api/recommend', { url })
      }

      console.log('Backend response:', response.data);

      if (response.data && response.data.recommendations) {
        setResults(response.data.recommendations)
      } else {
        setError('Invalid response from server')
      }
    } catch (err: any) {
      console.error('Error fetching recommendations:', err)
      const errorMessage = err.response?.data?.message || err.message || 'Failed to get recommendations. Please try again.'
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-extrabold text-text-primary sm:text-5xl mb-4 relative inline-block">
          <span className="relative z-10">SHL Assessment Recommender</span>
          <div className="absolute -bottom-2 left-0 w-full h-1 bg-gradient-to-r from-primary to-secondary rounded-full"></div>
        </h1>
        <p className="text-xl text-text-secondary max-w-3xl mx-auto">
          Find the most relevant SHL assessments for your job role or description
        </p>
      </div>

      <div className="max-w-3xl mx-auto mb-12">
        <div className="card glassmorphism animate-glow">
          <div className="mb-6">
            <div className="flex justify-center space-x-4 mb-6">
              <button
                type="button"
                className={`py-2 px-6 rounded-md transition-all duration-300 ${
                  inputMethod === 'text' 
                    ? 'bg-primary text-background shadow-md' 
                    : 'bg-card text-text-secondary hover:bg-card/80'
                }`}
                onClick={() => setInputMethod('text')}
              >
                Enter Text Query
              </button>
              <button
                type="button" 
                className={`py-2 px-6 rounded-md transition-all duration-300 ${
                  inputMethod === 'url' 
                    ? 'bg-primary text-background shadow-md' 
                    : 'bg-card text-text-secondary hover:bg-card/80'
                }`}
                onClick={() => setInputMethod('url')}
              >
                Enter Job URL
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            {inputMethod === 'text' ? (
              <div className="mb-4">
                <label htmlFor="query" className="label">
                  Job Description or Query
                </label>
                <textarea
                  id="query"
                  className="input h-36"
                  placeholder="Enter a job description or specific query..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  required
                />
              </div>
            ) : (
              <div className="mb-4">
                <label htmlFor="url" className="label">
                  Job Posting URL
                </label>
                <input
                  type="url"
                  id="url"
                  className="input"
                  placeholder="https://example.com/job-posting"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  required
                />
              </div>
            )}

            <div className="text-center">
              <button 
                type="submit" 
                className="btn-primary w-full sm:w-auto"
                disabled={loading}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-background" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Loading...
                  </span>
                ) : (
                  <span>Get Recommendations</span>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Results Section */}
      {error && (
        <div className="max-w-3xl mx-auto mb-8 p-4 bg-error/20 text-error border border-error/30 rounded-md">
          {error}
        </div>
      )}

      {results && results.length > 0 && (
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold mb-6 text-center">Recommendations</h2>
          <div className="grid gap-6 md:grid-cols-2">
            {results.map((assessment, index) => (
              <div key={index} className="card glassmorphism transition-all duration-300 hover:translate-y-[-5px]">
                <h3 className="text-xl font-semibold mb-3 text-primary flex items-center">
                  <span className="mr-2 inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary/20 text-primary text-xs">
                    {index + 1}
                  </span>
                  {assessment.name}
                </h3>
                
                <div className="space-y-2 mb-4 text-sm">
                  {assessment.test_type && (
                    <div className="flex items-start">
                      <span className="font-medium w-24 text-text-muted">Type:</span>
                      <span className="text-text-secondary">{assessment.test_type}</span>
                    </div>
                  )}
                  {assessment.duration && (
                    <div className="flex items-start">
                      <span className="font-medium w-24 text-text-muted">Duration:</span>
                      <span className="text-text-secondary">{assessment.duration} minutes</span>
                    </div>
                  )}
                  <div className="flex items-start">
                    <span className="font-medium w-24 text-text-muted">Remote:</span>
                    <span className="text-text-secondary">{assessment.remote_testing ? "Yes" : assessment.remote || "No"}</span>
                  </div>
                  <div className="flex items-start">
                    <span className="font-medium w-24 text-text-muted">Adaptive:</span>
                    <span className="text-text-secondary">{assessment.adaptive || "No"}</span>
                  </div>
                </div>
                
                {assessment.url ? (
                  <a 
                    href={assessment.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    onClick={() => {
                      console.log('Redirecting to:', assessment.url);
                    }}
                    className="btn-primary inline-block text-center"
                  >
                    View Assessment
                  </a>
                ) : (
                  <button
                    className="btn-primary inline-block text-center opacity-50"
                    disabled
                  >
                    No URL Available
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {results && results.length === 0 && (
        <div className="max-w-3xl mx-auto text-center p-6 bg-surface rounded-lg border border-border/30">
          <h2 className="text-xl font-medium text-text-primary">No assessments found</h2>
          <p className="text-text-muted mt-2">Try modifying your query to get better results.</p>
        </div>
      )}
    </div>
  )
} 