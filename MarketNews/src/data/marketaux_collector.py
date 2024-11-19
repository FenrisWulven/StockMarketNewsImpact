import requests
import json
import time
import os
import pandas as pd
from datetime import datetime

# Define base path
BASE_PATH = r'C:\Users\jbh\Desktop\CompTools\StockMarketNewsImpact\MarketNews'

def fetch_marketaux_news():
    # Parameters
    api_token = 'jvY5DqlTyacARjdScci70wqnLPynkDCkwEtJ5BTY'
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'META', 'AMZN', 'NVDA']
    year = '2024'
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    limit = 20  # Max articles per request (depends on your plan)
    max_requests = 300  # Adjust based on your plan
    request_count = 0
    all_articles = []

    # Iterate through companies and paginate results
    for symbol in symbols:
        print(f"Fetching news for {symbol}...")
        page = 1
        while request_count < max_requests:
            url = (
                f'https://api.marketaux.com/v1/news/all'
                f'?api_token={api_token}'
                f'&symbols={symbol}'
                f'&filter_entities=true'
                f'&min_match_score=40'
                f'&published_after={start_date}'
                f'&published_before={end_date}'
                f'&language=en'
                f'&group_similar=true'
                f'&limit={limit}'
                f'&page={page}'
            )
            response = requests.get(url)
            data = response.json()

            if 'data' in data:
                all_articles.extend(data['data'])
                if len(data['data']) < limit:  # No more articles for this symbol
                    break
                page += 1  # Go to the next page
                request_count += 1
                time.sleep(0.25)  # Rate limit handling
            else:
                break

    # Create necessary directories
    raw_dir = os.path.join(BASE_PATH, 'data', 'raw', 'marketaux')
    os.makedirs(raw_dir, exist_ok=True)

    # Save raw data
    raw_output_file = os.path.join(raw_dir, f'filtered_market_news_{year}.json')
    with open(raw_output_file, 'w') as f:
        json.dump(all_articles, f, indent=4)

    # Process the data into a more structured format
    processed_articles = []
    for article in all_articles:
        # Extract all symbols from entities
        entities = article.get('entities', [])
        if not entities:
            continue
            
        # Get datetime, skip article if not available
        published_at = article.get('published_at')
        if not published_at:
            continue
        
        # Create a news item for each entity that is an equity
        for entity in entities:
            if entity.get('type') == 'equity':
                news_item = {
                    'symbol': entity.get('symbol', ''),
                    'company_name': entity.get('name', ''),
                    'sentiment_score': entity.get('sentiment_score', None),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'snippet': article.get('snippet', ''),
                    'url': article.get('url', ''),
                    'datetime': published_at,
                    'source': article.get('source', ''),
                    'data_source': 'MarketAux'
                }
                processed_articles.append(news_item)

    # Create processed directory if it doesn't exist
    processed_dir = os.path.join(BASE_PATH, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Convert to DataFrame and save processed version
    df = pd.DataFrame(processed_articles)
    processed_output_file = os.path.join(processed_dir, f'marketaux_news_{year}.csv')
    df.to_csv(processed_output_file, index=False)

    print(f"\nTotal articles retrieved: {len(all_articles)}")
    print(f"Total processed articles: {len(processed_articles)}")
    print(f"Raw data saved to: {raw_output_file}")
    print(f"Processed data saved to: {processed_output_file}")

    # Print summary per company
    company_counts = df['symbol'].value_counts()
    print("\nArticles per company:")
    print(company_counts.to_string())

if __name__ == "__main__":
    fetch_marketaux_news()