
## Data Flow
1. Data Collection
   - `finnhub_collector.py`: Fetches financial news from Finnhub API
   - `marketaux_collector.py`: Fetches news from MarketAux API
   - Both save raw JSON data to `data/raw/`

2. Data Processing
   - Raw data is processed and converted to CSV format
   - Processed files are saved to `data/processed/`
   - `concat_data.py` combines data from both sources

3. Content Enrichment
   - `webScrape.py` fetches full article content
   - Tracks failed attempts and saves results to processed directory

4. Analysis
   - `sentiment_analyzer.py` creates similarity networks
   - Generates visualizations saved to `data/output/`

## Key Files
- `config.ini`: Contains API keys and settings
- `combined_news_data.csv`: Main dataset with all news articles
- `finnhub_scraped_articles.csv`: Articles with full content
- `article_similarity_network.png`: Visual network analysis
- `combined_article_distribution.png`: News source distribution

## Dependencies
- pandas
- requests
- beautifulsoup4
- networkx
- scikit-learn
- matplotlib
- seaborn
- finnhub-python
- fake-useragent

## Usage
1. Configure API keys in `config/config.ini`
2. Run data collectors:
   ```bash
   python src/data/finnhub_collector.py
   python src/data/marketaux_collector.py
   ```
3. Combine and process data:
   ```bash
   python src/data/concat_data.py
   ```
4. Scrape full articles:
   ```bash
   python src/data/webScrape.py
   ```
5. Run analysis:
   ```bash
   python src/analysis/sentiment_analyzer.py
   ```