import { NextRequest, NextResponse } from 'next/server'

// Backend API URL - Uses Render URL in production, localhost in development
const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'https://shl-assessment-1-jk29.onrender.com/api/recommend'
    : 'http://localhost:5000/api/recommend');

// Get recommendations from the Python backend
async function getRecommendations(query: string | null, url: string | null) {
  try {
    console.log(`Calling backend API with ${query ? 'query' : 'url'}: ${query || url}`);
    
    // Make the actual API call to the Python backend
    const response = await fetch(BACKEND_API_URL, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(query ? { query } : { url }),
      cache: 'no-store',
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({
        message: `HTTP error ${response.status}: ${response.statusText}`
      }));
      console.error('Backend API error:', errorData);
      throw new Error(errorData.message || `HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    
    // Log the URLs from the response data
    if (data.recommendations) {
      console.log('Received recommendations with URLs:');
      data.recommendations.forEach((rec: any, index: number) => {
        console.log(`[${index + 1}] ${rec.name}: URL = ${rec.url}`);
      });
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, url } = body;

    if (!query && !url) {
      return NextResponse.json(
        { error: 'Missing required field', message: 'Please provide either query or url parameter' },
        { status: 400 }
      );
    }

    const data = await getRecommendations(query, url);
    
    // Process URLs to ensure they're valid
    if (data.recommendations) {
      data.recommendations = data.recommendations.map((rec: any) => {
        // Ensure URL is properly formatted
        if (rec.url && typeof rec.url === 'string') {
          // If URL doesn't start with http:// or https://, add https://
          if (!rec.url.match(/^https?:\/\//)) {
            rec.url = 'https://' + rec.url;
          }
          
          console.log(`Processed URL for ${rec.name}: ${rec.url}`);
        } else {
          console.log(`No valid URL for ${rec.name}`);
        }
        
        return rec;
      });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error processing request:', error);
    return NextResponse.json(
      { 
        error: 'Error processing request', 
        message: error instanceof Error ? error.message : 'An unexpected error occurred',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
} 
