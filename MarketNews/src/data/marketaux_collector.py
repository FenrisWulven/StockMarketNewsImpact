import requests
import json
import time
import os
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Define base path
BASE_PATH = r'C:\Users\jbhan\Desktop\StockMarketNewsImpact\MarketNews\data\raw'

def get_trading_days(start_date, end_date):
    """Get NYSE trading days between start and end date"""
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.schedule(start_date=start_date, end_date=end_date)
    return trading_days.index.date

def load_existing_uuids(year):
    """Load UUIDs of existing articles from the previous JSON file"""
    # Check the original file for existing UUIDs
    existing_file = Path(BASE_PATH) / 'marketaux' / f'filtered_market_news_{year}.json'
    existing_uuids = set()
    
    if existing_file.exists():
        try:
            with open(existing_file, 'r') as f:
                existing_articles = json.load(f)
            existing_uuids.update({article.get('uuid') for article in existing_articles if article.get('uuid')})
        except Exception as e:
            print(f"Error loading existing articles: {e}")
    
    # Also check the _new file if it exists
    new_file = Path(BASE_PATH) / 'marketaux' / f'filtered_market_news_{year}_new.json'
    if new_file.exists():
        try:
            with open(new_file, 'r') as f:
                new_articles = json.load(f)
            existing_uuids.update({article.get('uuid') for article in new_articles if article.get('uuid')})
        except Exception as e:
            print(f"Error loading new articles: {e}")
    
    return existing_uuids

def collect_news(api_key, symbols, start_date, end_date):
    """
    Collect news articles for given symbols between dates using pagination
    """
    base_url = "https://api.marketaux.com/v1/news/all"
    all_news = []
    collected_uuids = set()  # Track all collected UUIDs across days
    
    # Load existing UUIDs
    year = start_date[:4]
    existing_uuids = load_existing_uuids(year)
    print(f"Loaded {len(existing_uuids)} existing article UUIDs")
    
    # Get trading days
    trading_days = get_trading_days(start_date, end_date)
    print(f"Found {len(trading_days)} trading days between {start_date} and {end_date}")
    
    for date in tqdm(trading_days, desc="Collecting daily news"):
        formatted_date = date.strftime('%Y-%m-%d')
        daily_articles = []
        company_counts = {symbol: 0 for symbol in symbols}
        page = 1
        
        while True:
            params = {
                'api_token': api_key,
                'symbols': ','.join(symbols),
                'filter_entities': 'true',
                'min_match_score': 80,
                'language': 'en',
                'date': formatted_date,
                'limit': 20,
                'page': page
            }
            
            try:
                response = requests.get(base_url, params=params)
                
                if response.status_code == 429:
                    print(f"\nRate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data['data']:
                    break  # No more articles for this day
                
                # Process articles
                for article in data['data']:
                    article_uuid = article.get('uuid')
                    
                    # Skip if article UUID is in existing or already collected
                    if (article_uuid in existing_uuids or 
                        article_uuid in collected_uuids):
                        continue
                    
                    entities = article.get('entities', [])
                    relevant_article = False
                    
                    # Check which companies are mentioned with sufficient relevance
                    for entity in entities:
                        if (entity.get('type') == 'equity' and 
                            entity.get('symbol') in symbols and 
                            entity.get('match_score', 0) >= 90):
                            
                            company_counts[entity.get('symbol')] += 1
                            relevant_article = True
                    
                    if relevant_article:
                        daily_articles.append(article)
                        collected_uuids.add(article_uuid)  # Add to collected UUIDs
                
                print(f"\rDate: {formatted_date} - Page: {page} - Articles found: {len(daily_articles)} - Counts:", end='')
                for symbol, count in company_counts.items():
                    print(f" {symbol}: {count}", end='')
                
                # Break conditions
                if len(data['data']) < 20 or page >= 10:  # Added page limit
                    break
                
                page += 1
                time.sleep(0.01)  # Increased delay between requests
                
            except Exception as e:
                print(f"\nError collecting news for {formatted_date}: {str(e)}")
                if 'rate limit' in str(e).lower():
                    print("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                break
        
        # Add daily articles to overall collection
        all_news.extend(daily_articles)
        
        print(f"\nCollected {len(daily_articles)} unique articles for {formatted_date}")
        print("Final article counts:")
        for company, count in company_counts.items():
            print(f"{company}: {count}")
        print()
    
    print(f"\nTotal unique articles collected: {len(all_news)}")
    print(f"Skipped {len(existing_uuids)} existing articles")
    
    return all_news

def print_unique_article_stats(articles):
    """Print statistics about unique articles per company"""
    company_articles = {}
    all_uuids = set()
    
    for article in articles:
        article_uuid = article.get('uuid')
        if article_uuid:
            all_uuids.add(article_uuid)
        
        entities = article.get('entities', [])
        
        for entity in entities:
            if entity.get('type') == 'equity':
                symbol = entity.get('symbol')
                if symbol not in company_articles:
                    company_articles[symbol] = set()
                if article_uuid:
                    company_articles[symbol].add(article_uuid)
    
    print("\nUnique articles per company:")
    for symbol, article_uuids in sorted(company_articles.items()):
        print(f"{symbol}: {len(article_uuids)} unique articles")
    print(f"Total unique articles across all companies: {len(all_uuids)}")

def fetch_marketaux_news():
    # Parameters
    api_token = 'jvY5DqlTyacARjdScci70wqnLPynkDCkwEtJ5BTY'
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'META', 'AMZN', 'NVDA']
    year = '2022'
    start_date = f'{year}-02-06'
    end_date = f'{year}-12-31'  # Extended date range
    
    # Collect news articles (now returns a list)
    news_articles = collect_news(api_token, symbols, start_date, end_date)

    if not news_articles:  # Check if the list is empty
        print("No articles collected. Exiting.")
        return

    # Create necessary directories
    raw_dir = os.path.join(BASE_PATH, 'marketaux')
    os.makedirs(raw_dir, exist_ok=True)

    # Add this before saving to file
    print_unique_article_stats(news_articles)
    
    # Format the date range for the filename
    date_range = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    new_file = os.path.join(raw_dir, f'filtered_market_news_{year}_{date_range}.json')
    
    # Save directly to the file
    with open(new_file, 'w') as f:
        json.dump(news_articles, f, indent=4)

    # Process the data into a more structured format
    processed_articles = []
    for article in news_articles:  # Use the list directly instead of converting to dict
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
    processed_dir = os.path.join(BASE_PATH, '..', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Convert to DataFrame and save processed version with _new suffix
    df = pd.DataFrame(processed_articles)
    
    if len(df) == 0:    
        print("No processed articles. Exiting.")
        return
        
    # Update processed filename to use date range instead of timestamp
    processed_output_file = os.path.join(processed_dir, f'marketaux_news_{year}_{date_range}.csv')
    df.to_csv(processed_output_file, index=False)

    print(f"\nTotal articles retrieved: {len(news_articles)}")
    print(f"Total processed articles: {len(processed_articles)}")
    print(f"Raw data saved to: {new_file}")
    print(f"Processed data saved to: {processed_output_file}")

    if 'symbol' in df.columns:
        # Print summary per company
        company_counts = df['symbol'].value_counts()
        print("\nArticles per company:")
        print(company_counts.to_string())   
    else:
        print("\nNo symbol column found in processed data")

if __name__ == "__main__":
    fetch_marketaux_news()