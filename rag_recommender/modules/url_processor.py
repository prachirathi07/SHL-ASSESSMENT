"""
URL Processing Module for SHL Assessment Recommendation System
This module handles fetching and parsing job descriptions from URLs.
"""
import requests
from bs4 import BeautifulSoup
import logging
import re
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def is_valid_url(url: str) -> bool:
    """
    Check if a given string is a valid URL.
    
    Args:
        url: String to check
        
    Returns:
        Boolean indicating if the string is a valid URL
    """
    url_pattern = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))

def clean_job_description(text: str) -> str:
    """
    Clean job description text by removing unnecessary elements.
    
    Args:
        text: Raw job description text
        
    Returns:
        Cleaned job description
    """
    # Remove multiple newlines, tabs, and spaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common LinkedIn UI text elements
    ui_elements = [
        r'Apply', r'Save', r'Report this job', r'See who .* has hired for this role',
        r'Join or sign in to find your next job', r'Join to apply for the .* role at .*',
        r'Not you\? Remove photo', r'First name', r'Last name', r'Email', r'Password',
        r'By clicking Agree & Join, you agree to the LinkedIn User Agreement',
        r'Privacy Policy', r'Cookie Policy', r'Continue', r'Agree & Join',
        r'You may also apply directly on company website', r'Security verification',
        r'Already on LinkedIn\? Sign in', r'Sign in Welcome back', r'Welcome back',
        r'Sign in or', r'New to LinkedIn\? Join now', r'Forgot password\?',
        r'Set alert', r'Similar jobs', r'See more jobs like this', r'Show fewer jobs like this',
        r'People also viewed', r'Am I a good fit for this job\?', r'Tailor my resume',
        r'Sign in to access AI-powered advices', r'Get AI-powered advice',
        r'Use AI to assess how you fit', r'Show more', r'Show less',
        r'Get notified when a new job is posted', r'Similar Searches',
        r'Referrals increase your chances of interviewing',
        r'This range is provided by .* Your actual pay will be based on your skills and experience — talk with your recruiter to learn more',
        r'See who you know'
    ]
    
    # Remove each UI element
    for element in ui_elements:
        cleaned = re.sub(element, '', cleaned, flags=re.IGNORECASE)
    
    # Remove sections with open jobs counts
    cleaned = re.sub(r'\d+ open jobs', '', cleaned)
    
    # Remove duplicate skills sections (common in LinkedIn job posts)
    if "Skills:" in cleaned and cleaned.count("Skills:") > 1:
        # Only keep the first skills section
        parts = cleaned.split("Skills:", 1)
        skills_parts = parts[1].split("Skills:", 1)
        cleaned = parts[0] + "Skills:" + skills_parts[0]
    
    # Remove repeated job titles and locations
    job_title_pattern = re.compile(r'([A-Za-z\s&\-\/]+) \1+')
    cleaned = job_title_pattern.sub(r'\1', cleaned)
    
    # Remove LinkedIn login prompts and UI instructions
    cleaned = re.sub(r'Sign in to.*', '', cleaned)
    cleaned = re.sub(r'Join to.*', '', cleaned)
    
    # Final clean-up of multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def extract_linkedin_job_content(soup: BeautifulSoup) -> str:
    """
    Specialized extraction for LinkedIn job postings.
    
    Args:
        soup: BeautifulSoup object of the LinkedIn page
        
    Returns:
        Extracted job description
    """
    # Try to find the job description section
    job_description = ""
    
    # Look for job description container
    description_div = soup.select_one('[class*="description"]')
    if description_div:
        job_description = description_div.get_text(separator=' ', strip=True)
    
    # If not found, try alternate approach for LinkedIn's structure
    if not job_description or len(job_description) < 50:
        # Look for common LinkedIn job posting patterns
        main_content = soup.select_one('main')
        if main_content:
            job_description = main_content.get_text(separator=' ', strip=True)
        
        # If still not found, extract from body with some filtering
        if not job_description or len(job_description) < 50:
            # Extract all text blocks and filter out short ones
            blocks = []
            for p in soup.find_all(['p', 'li', 'div']):
                text = p.get_text(strip=True)
                if len(text) > 50:  # Only consider substantial text blocks
                    blocks.append(text)
            
            job_description = ' '.join(blocks)
    
    # Clean up the text
    job_description = clean_job_description(job_description)
    
    return job_description

def is_linkedin_url(url: str) -> bool:
    """Check if the URL is from LinkedIn"""
    return "linkedin.com" in url.lower()

def fetch_job_description(url: str) -> Tuple[bool, Optional[str]]:
    """
    Fetch and parse job description from a URL.
    
    Args:
        url: URL to fetch job description from
        
    Returns:
        Tuple containing success flag and job description text (if successful)
    """
    try:
        # Check if URL is valid
        if not is_valid_url(url):
            logging.error(f"Invalid URL format: {url}")
            return False, "Invalid URL format. Please provide a valid HTTP or HTTPS URL."
        
        # Fetch the page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Special handling for LinkedIn
        if is_linkedin_url(url):
            logging.info("Detected LinkedIn URL, using specialized extraction")
            job_description = extract_linkedin_job_content(soup)
            # For debugging purposes, print the first part of the processed description
            logging.debug(f"LinkedIn extraction results (first 200 chars): {job_description[:200]}")
        else:
            # Extract text from main content elements where job descriptions are typically found
            content_selectors = [
                'article', 'main', '.job-description', '.description', 
                '#job-description', '[class*="job"]', '[class*="description"]',
                '.about-job', '.details', '.content'
            ]
            
            content = []
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        content.append(element.get_text(separator=' ', strip=True))
            
            # If we found specific content areas, use them
            if content:
                job_description = ' '.join(content)
            else:
                # Fall back to extracting paragraphs if specific sections aren't found
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 100]
                job_description = ' '.join(paragraphs)
            
            # Clean the text
            job_description = re.sub(r'\s+', ' ', job_description).strip()
        
        # Check if we actually got meaningful content
        if len(job_description) < 50:
            logging.warning(f"Content from URL seems too short: {len(job_description)} chars")
            return False, "Could not extract meaningful job description from the URL. The content seems too short or in an unsupported format."
        
        logging.info(f"Successfully extracted {len(job_description)} chars from URL")
        return True, job_description
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {str(e)}")
        return False, f"Error fetching URL: {str(e)}"
    
    except Exception as e:
        logging.error(f"Error processing URL {url}: {str(e)}")
        return False, f"Error processing URL: {str(e)}"

if __name__ == "__main__":
    test_url = "https://www.example.com/job-posting"
    success, content = fetch_job_description(test_url)
    if success:
        print(f"Content length: {len(content)}")
        print(content[:200] + "...")
    else:
        print(f"Error: {content}") 