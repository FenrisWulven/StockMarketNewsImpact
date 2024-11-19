import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from tqdm import tqdm
import os
from fake_useragent import UserAgent
import random
import json
from datetime import datetime

# Define base path
BASE_PATH = r'C:\Users\jbh\Desktop\CompTools\StockMarketNewsImpact\MarketNews'

# Define failure tracking file with updated path
FAILED_SCRAPES_FILE = os.path.join(BASE_PATH, 'data', 'processed', 'failed_scrapes.json')

def load_failed_scrapes():
    """Load the existing failed scrapes"""
    if os.path.exists(FAILED_SCRAPES_FILE):
        with open(FAILED_SCRAPES_FILE, 'r') as f:
            return json.load(f)
    return {'failed_urls': [], 'failure_reasons': {}, 'last_updated': None}

def save_failed_scrapes(failed_data):
    """Save the failed scrapes with timestamp"""
    failed_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(FAILED_SCRAPES_FILE, 'w') as f:
        json.dump(failed_data, f, indent=4)

def get_actual_url(finnhub_url):
    """Fetch the actual URL from Finnhub's redirect endpoint"""
    try:
        response = requests.get(finnhub_url, allow_redirects=False)
        if response.status_code == 302:  # Check for redirect
            return response.headers.get('Location', finnhub_url)
        return finnhub_url
    except Exception as e:
        return f"Error: {str(e)}"

def get_random_headers():
    """Generate random headers to avoid detection"""
    ua = UserAgent()
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def is_cookie_content(text):
    """Check if text is related to cookie consent or privacy policy"""
    cookie_keywords = [
        'cookie', 'privacy', 'gdpr', 'consent', 'acceptér', 'privatlivs',
        'vi, yahoo', 'yahoo-familien', 'websites og apps', 'privatlivspolitik',
        'accept all', 'reject all', 'manage settings'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in cookie_keywords)

def extract_article_content(url):
    """Extract article content from a given URL"""
    try:
        time.sleep(random.uniform(1, 3))
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove common cookie/privacy elements
        for element in soup.find_all(['div', 'section', 'iframe'], class_=lambda x: x and any(keyword in str(x).lower() for keyword in ['cookie', 'consent', 'privacy', 'gdpr'])):
            element.decompose()
        
        # Remove other unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        content = ""
        
        # Strategy 1: Look for article tag
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50 and not is_cookie_content(p.get_text()))
        
        # Strategy 2: Look for main content div
        if not content:
            main_content = soup.find(['main', 'div'], class_=lambda x: x and any(word in x.lower() for word in ['content', 'article', 'story', 'body']))
            if main_content:
                paragraphs = main_content.find_all('p')
                content = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50 and not is_cookie_content(p.get_text()))
        
        # Strategy 3: Look for any substantial paragraphs
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text(strip=True) for p in paragraphs 
                             if len(p.get_text(strip=True)) > 50 
                             and not is_cookie_content(p.get_text()))
        
        # Verify content is not just cookie/privacy text
        if content and not is_cookie_content(content[:200]):
            return content
        return "Content extraction failed or only found cookie/privacy content"
    
    except Exception as e:
        return f"Error: {str(e)}"

def scrape_finnhub_articles():
    # Load the combined news data from processed directory
    df = pd.read_csv(os.path.join(BASE_PATH, 'data', 'processed', 'combined_news_data.csv'))
    
    # Load existing failed scrapes
    failed_scrapes = load_failed_scrapes()
    
    # Filter Finnhub articles
    finnhub_df = df[df['data_source'] == 'Finnhub'].head(100).copy()
    
    print("Resolving Finnhub URLs and scraping content...")
    
    actual_urls = []
    article_contents = []
    new_failures = {'failed_urls': [], 'failure_reasons': {}}
    
    for url in tqdm(finnhub_df['url'], desc="Scraping articles"):
        # Get actual URL
        actual_url = get_actual_url(url)
        if actual_url.startswith('Error:'):
            new_failures['failed_urls'].append(url)
            new_failures['failure_reasons'][url] = {
                'error_type': 'redirect_error',
                'error_message': actual_url,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            actual_urls.append(url)  # Keep original URL
            article_contents.append("Failed to resolve URL")
            continue
            
        actual_urls.append(actual_url)
        
        # Scrape content
        content = extract_article_content(actual_url)
        
        # Track failures
        if content.startswith('Error:') or content.startswith('Content extraction failed'):
            new_failures['failed_urls'].append(actual_url)
            new_failures['failure_reasons'][actual_url] = {
                'error_type': 'scraping_error',
                'error_message': content,
                'original_url': url,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        article_contents.append(content)
    
    # Update failed scrapes data and save to processed directory
    failed_scrapes['failed_urls'].extend(new_failures['failed_urls'])
    failed_scrapes['failure_reasons'].update(new_failures['failure_reasons'])
    save_failed_scrapes(failed_scrapes)
    
    # Create failure summary and save to processed directory
    failure_summary = pd.DataFrame([
        {
            'original_url': url if 'original_url' not in info else info['original_url'],
            'failed_url': url,
            'error_type': info['error_type'],
            'error_message': info['error_message'],
            'timestamp': info['timestamp']
        }
        for url, info in new_failures['failure_reasons'].items()
    ])
    
    if not failure_summary.empty:
        failure_summary.to_csv(os.path.join(BASE_PATH, 'data', 'processed', 'recent_failures.csv'), index=False)
    
    # Add columns to DataFrame
    finnhub_df['actual_url'] = actual_urls
    finnhub_df['article_content'] = article_contents
    
    # Create processed directory if it doesn't exist
    processed_dir = os.path.join(BASE_PATH, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save results to processed directory
    output_file = os.path.join(processed_dir, 'finnhub_scraped_articles.csv')
    finnhub_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nScraping Summary:")
    print(f"Total articles processed: {len(finnhub_df)}")
    successful_articles = finnhub_df[
        (finnhub_df['article_content'].str.len() > 100) & 
        (~finnhub_df['article_content'].str.startswith('Error')) &
        (~finnhub_df['article_content'].str.startswith('Content extraction failed'))
    ]
    print(f"Successfully scraped articles: {len(successful_articles)}")
    print(f"Failed articles: {len(finnhub_df) - len(successful_articles)}")
    
    # Print failure details
    if not failure_summary.empty:
        print("\nFailure Summary:")
        print(f"Total new failures: {len(failure_summary)}")
        print("\nFailure types:")
        print(failure_summary['error_type'].value_counts())
        
    return finnhub_df

if __name__ == "__main__":
    scraped_df = scrape_finnhub_articles() 