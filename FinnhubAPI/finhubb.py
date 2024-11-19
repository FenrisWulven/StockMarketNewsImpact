import finnhub
import json
import time
from datetime import datetime, timedelta
import configparser

def fetch_finnhub_news():
    # Initialize the config parser
    config = configparser.ConfigParser()
    config.read(r'C:\Users\jbh\Desktop\CompTools\StockMarketNewsImpact\FinnhubAPI\config.ini')

    # Retrieve the API key from the config file
    api_key = config.get('finnhub', 'api_key')

    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key=api_key)

    # Parameters
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'META', 'AMZN', 'NVDA']
    year = '2024'
    start_date = f'{year}-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')  # Current date

    all_articles = []

    def fetch_company_news(symbol, from_date, to_date):
        """Fetch news for a company with rate limiting and error handling"""
        try:
            news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            time.sleep(1)  # Rate limiting - 1 second delay between requests
            return news
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            time.sleep(60)  # If error occurs, wait longer
            return []

    def convert_timestamp(timestamp):
        """Convert timestamp to datetime string, handling both seconds and milliseconds formats"""
        try:
            # Try converting directly (assuming seconds)
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (OSError, ValueError):
            try:
                # Try converting from milliseconds
                return datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
            except:
                # Return current time if conversion fails
                print(f"Warning: Invalid timestamp {timestamp}, using current time")
                return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Fetch news for each company
    for symbol in symbols:
        print(f"Fetching news for {symbol}...")
        
        # Fetch news in smaller date chunks to handle rate limits
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date < end_datetime:
            # Create 7-day chunks
            chunk_end = min(current_date + timedelta(days=7), end_datetime)
            
            # Format dates for API
            from_date = current_date.strftime('%Y-%m-%d')
            to_date = chunk_end.strftime('%Y-%m-%d')
            
            # Fetch news for this chunk
            company_news = fetch_company_news(symbol, from_date, to_date)
            
            # Process and store the news
            for article in company_news:
                article_data = {
                    'symbol': symbol,
                    'datetime': convert_timestamp(article['datetime']),
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'id': article.get('id', '')
                }
                all_articles.append(article_data)
            
            print(f"Fetched {len(company_news)} articles for {symbol} from {from_date} to {to_date}")
            current_date = chunk_end + timedelta(days=1)

    # Save results to a JSON file
    output_file = f'finnhub_news_{year}.json'
    with open(output_file, 'w') as f:
        json.dump(all_articles, f, indent=4)

    print(f"\nTotal articles retrieved: {len(all_articles)}")
    print(f"Data saved to {output_file}")

    # Print summary per company
    company_counts = {}
    for article in all_articles:
        symbol = article['symbol']
        company_counts[symbol] = company_counts.get(symbol, 0) + 1

    print("\nArticles per company:")
    for symbol, count in company_counts.items():
        print(f"{symbol}: {count} articles")

if __name__ == "__main__":
    fetch_finnhub_news()
