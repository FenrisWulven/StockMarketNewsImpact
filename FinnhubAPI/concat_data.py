import pandas as pd
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_news_data():
    # Find all JSON files for both sources
    marketaux_files = glob.glob('filtered_market_news_*.json')
    finnhub_files = glob.glob('finnhub_news_*.json')
    
    all_news = []
    
    # Process MarketAux files
    for file_path in marketaux_files:
        print(f"Processing MarketAux file: {file_path}...")
        with open(file_path, 'r') as f:
            articles = json.load(f)
            
            # Process each MarketAux article
            for article in articles:
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
                            'datetime': published_at,  # ISO format from MarketAux
                            'source': article.get('source', ''),
                            'data_source': 'MarketAux'
                        }
                        all_news.append(news_item)
    
    # Process Finnhub files
    for file_path in finnhub_files:
        print(f"Processing Finnhub file: {file_path}...")
        with open(file_path, 'r') as f:
            articles = json.load(f)
            
            # Process each Finnhub article
            for article in articles:
                # Convert Finnhub datetime to ISO format
                dt = article.get('datetime', '')
                try:
                    # Parse the datetime string and convert to ISO format
                    dt_obj = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                    iso_dt = dt_obj.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
                except:
                    print(f"Warning: Could not parse datetime {dt}")
                    continue

                news_item = {
                    'symbol': article.get('symbol', ''),
                    'company_name': '',  # Finnhub doesn't provide company name
                    'sentiment_score': None,  # Finnhub doesn't provide sentiment
                    'title': article.get('headline', ''),
                    'description': article.get('summary', ''),
                    'snippet': article.get('summary', '')[:200] if article.get('summary') else '',
                    'url': article.get('url', ''),
                    'datetime': iso_dt,  # Converted to ISO format
                    'source': article.get('source', ''),
                    'data_source': 'Finnhub'
                }
                all_news.append(news_item)
    
    if not all_news:
        raise ValueError("No valid articles found in the JSON files")
    
    # Create DataFrame
    df = pd.DataFrame(all_news)
    
    # Convert datetime string to datetime object with error handling
    try:
        # Use format='ISO8601' to handle ISO format dates
        df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
    except Exception as e:
        print(f"Error converting datetime: {e}")
        print("Sample of datetime values:", df['datetime'].head())
        raise
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Save to CSV format
    df.to_csv('combined_news_data.csv', index=False)
    
    print(f"\nProcessed {len(df)} articles")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Create summary DataFrames
    symbol_summary = df.groupby(['symbol', 'data_source']).size().reset_index(name='Article Count')
    
    # Print summary
    print("\nArticles per symbol and source:")
    print(symbol_summary.to_string(index=False))
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Create grouped bar plot
    ax = sns.barplot(
        data=symbol_summary,
        x='symbol',
        y='Article Count',
        hue='data_source',
        palette=['skyblue', 'lightgreen']
    )
    
    # Customize the plot
    plt.title('Distribution of Articles Across Companies and Sources', pad=20, size=14)
    plt.xlabel('Company Symbol', size=12)
    plt.ylabel('Number of Articles', size=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('combined_article_distribution.png')
    plt.close()
    
    return df

if __name__ == "__main__":
    try:
        df = load_and_process_news_data()
    except Exception as e:
        print(f"Error processing news data: {e}")
