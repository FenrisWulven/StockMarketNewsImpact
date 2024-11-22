import praw
import pandas as pd
import datetime
import time
import os
import logging
from dotenv import load_dotenv
from textblob import TextBlob
import prawcore
import praw.exceptions
import random
import re  # For efficient keyword matching

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(filename="reddit_data_collection.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Add console logging for real-time feedback
logging.getLogger().addHandler(logging.StreamHandler())
# logger = logging.getLogger()
# if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
#     logger.addHandler(logging.StreamHandler())

# Initialize Reddit API with environment variables
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
    ratelimit_seconds=600
)

# Variables
limit = 200  # Number of posts to fetch for each keyword


company_keywords = {
    # "Apple": [
    #     "Apple", "$AAPL", "AAPL", "MacBook", "iPhone", "Apple Inc", 
    #     "Tim Cook", "iOS", "iPad", "Apple Watch", "Mac Studio", "Siri", "AirPods", "Mac Mini", 
    #     "Mac Pro", "M1", "M2", "M3"
    # ],
    # "Microsoft": [
    #     "Microsoft", "$MSFT", "MSFT", "Windows", "Azure", "Office 365", 
    #     "Satya Nadella", "Xbox", "Surface", "Microsoft Teams", "LinkedIn",
    #     "OpenAI", "Chat-GPT", "4o1", "GPT-3", "GPT-4"
    # ],
    # "Nvidia": [
    #     "Nvidia", "$NVDA", "NVDA", "GeForce", "RTX", "GPU", "Super Computer", 
    #     "Jensen Huang", "Nvidia AI", "CUDA", "Hopper GPU", "Omniverse"
    # ],
    "Tesla": [
        #"Tesla", "$TSLA", "TSLA", "Elon Musk", "Elon", "Musk",
        "Model 3", "Model S", "FSD", 
        "PowerWall", "Megapack", "Cybertruck", "Model Y", "Tesla Semi", "Gigafactory"
    ],
    "Amazon": [
        "Amazon", "$AMZN", "AMZN", "AWS", "Prime", "Amazon Web Services", 
        "Twitch", "Alexa", "Kindle", "Prime Video", "Andy Jassy", "Amazon Go",
        "Amazon Fresh", "Amazon Robotics", "Jeff Bezos", "Bezos"
    ],
    "Google": [
        "Google", "$GOOGL", "GOOGL", "Alphabet", "YouTube", "Google Cloud", 
        "Sundar Pichai", "Pixel", "Waymo", "Nest", "Bard AI", "Google Ads", "Gemini"
    ],
    "Meta": [
        "Meta", "$META", "META", "Facebook", "Instagram", "WhatsApp", "Oculus", 
        "Mark Zuckerberg", "Zuckerberg", "Zuck", "Horizon Worlds", "Threads", "Meta Quest", 
        "Reels", "Metaverse", "LLaMa"
    ]
}

financial_terms = [
    "stock", "shares", "market", "earnings", "dividends", "revenue", "event", "news",
    "growth", "forecast", "profit", "loss", "valuation", "price target", "products", "ecosystem",
    "buy", "sell", "bullish", "bearish", "EPS", "PE ratio", "market cap", "market share", "innovation",
    "short interest", "institutional ownership", "insider trading", "SEC filing", "buyback", "split",
    "10-K", "10-Q", "8-K", "annual report", "quarterly report", "balance sheet", "income",
    "income statement", "cash flow", "financials", "fundamentals", "technical analysis", 
    "candlestick", "moving average", "RSI", "MACD", "Bollinger Bands", "support level",
    "resistance level", "options", "calls", "puts", "strike price", "expiration date",
    "implied volatility", "open interest", "volume", "short squeeze", "gamma squeeze",
    "upside potential", "downside risk", "bull case", "bear case", "long-term", "short-term",
    "day trading", "swing trading", "value investing", "growth investing", "dividend investing",
    "up", "down", "trending", "consolidating", "sideways", "breakout", "pullback", "reversal",
    "correction", "crash", "rally", "bubble", "recession", "inflation", "deflation", "unemployment",
    "GDP", "interest rates", "Federal Reserve", "FOMC", "monetary policy", "fiscal policy",
    "stimulus", "infrastructure bill", "tax plan", "capital gains", "inflation rate", "CPI", "PPI",
    "unemployment rate", "jobless claims", "retail sales", "industrial production", "housing starts",
    "building permits", "consumer sentiment", "business sentiment", "manufacturing", "services",
    "technology", "healthcare", "energy", "financials", "consumer discretionary", "consumer staples",
    "utilities", "real estate", "materials", "industrials", "communication services", "technology",
    "consumer cyclical", "consumer defensive", "basic materials", "communication services", "energy",
    "financial services", "healthcare", "industrials", "real estate", "technology", "utilities",
    "small-cap", "mid-cap", "large-cap", "growth stocks", "value stocks", "dividend stocks",
    "cyclical stocks", "defensive stocks", "blue-chip stocks", "penny stocks", "meme stocks",
    "short squeeze stocks", "high short interest stocks", "high volume stocks", "low volume stocks", "cap"
]

financial_pattern = re.compile('|'.join(financial_terms), re.IGNORECASE)

# Define time range for 2018-2022 in UNIX timestamps
start_timestamp = int(datetime.datetime(2021, 12, 1).timestamp())
end_timestamp = int(datetime.datetime(2024, 10, 31, 23, 59, 59).timestamp())

# Function to get the top 10 comments for a post with a reduced delay
def get_top_comments(post, limit=10):
    comments = []
    try:
        safe_request(post.comments.replace_more, limit=0)
        for comment in post.comments[:limit]:
            comments.append(comment.body)
            time.sleep(0.1)
    except Exception as e:
        logging.warning(f"Error fetching comments: {e}")
    return comments

# Function to calculate sentiment polarity and subjectivity
def analyze_text(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Function to save data incrementally to CSV
def save_to_csv(df, company_name):
    try:
        csv_name = f"reddit_stock_data_{company_name}.csv"
        file_exists = os.path.isfile(csv_name)
        df.to_csv(csv_name, mode='a', index=False, header=not file_exists)
        logging.info(f"Data appended to CSV for {company_name} as {csv_name}!")
    except Exception as e:
        logging.error(f"Error saving to CSV for {company_name}: {e}")
        raise

# Function to save remaining keywords to a checkpoint file
def save_remaining_keywords(remaining_keywords):
    with open("remaining_keywords_checkpoint.txt", "w") as f:
        for keyword in remaining_keywords:
            f.write(f"{keyword}\n")

def safe_request(callable_fn, *args, **kwargs):
    retry_attempts = 0
    while retry_attempts < 10:  # Max retries to avoid infinite loop
        try:
            return callable_fn(*args, **kwargs)
        except praw.exceptions.RedditAPIException as e:
            for subexception in e.items:
                if "RATELIMIT" in subexception.error_type:
                    wait_time = extract_wait_time(subexception.message)
                    if wait_time:
                        logging.warning(f"Rate limit retry #{retry_attempts + 1} for '{args}' in {wait_time} seconds.")
                        #print(f"Rate limit hit. Sleeping for {wait_time} seconds...")
                        time.sleep(wait_time + 1)
                    else:
                        logging.warning("Unknown rate limit. Sleeping for 60 seconds...")
                        #print("Unknown rate limit. Sleeping for 60 seconds...")
                        time.sleep(60)
                    retry_attempts += 1
                else:
                    logging.error(f"RedditAPIException (Non-Rate Limit) for {subexception.error_type}: {subexception.message}")
                    raise  # Reraise other exceptions
        except Exception as e:
            logging.error(f"Unexpected error during API request: {e}")
            raise
    logging.error("Max retries exceeded. Terminating request.")
    raise Exception("Max retries exceeded.")

def extract_wait_time(message):
    """Extract wait time in seconds from rate limit message."""
    match = re.search(r"(\d+) (?:minute|second)s?", message)
    if match:
        wait_time = int(match.group(1))
        if "minute" in message:
            wait_time *= 60
        return wait_time
    return None

# Function to fetch posts based on keywords across all of Reddit
def fetch_keyword_data(company, keywords, limit=limit):
    data = []
    
    for keyword in keywords:
        retry_attempts = 0
        while retry_attempts < 5:
            try:
                logging.info(f"Fetching posts for keyword: {keyword} in company: {company}")
                # Search for posts containing the keyword, sorted by top score
                for post in safe_request(reddit.subreddit("all").search, keyword, sort="top", limit=limit):
                #for post in reddit.subreddit("all").search(keyword, sort="top", limit=limit):
                    if start_timestamp <= post.created_utc <= end_timestamp:
                        # also comments
                        content = (post.title or '') + ' ' + (post.selftext or '') 
                        top_comments = get_top_comments(post, limit=10)
                        content += ' '.join(top_comments)
                        
                        # Filter using financial terms
                        if financial_pattern.search(content):
                            polarity, subjectivity = analyze_text(post.selftext)
                            post_data = {
                                "subreddit": post.subreddit.display_name,
                                "title": post.title,
                                "timestamp": datetime.datetime.fromtimestamp(post.created_utc),
                                "content": post.selftext,
                                "score": post.score,
                                "num_comments": post.num_comments,
                                "author": post.author.name if post.author else "N/A",
                                "sentiment": polarity,
                                "subjectivity": subjectivity,
                                "top_comments": get_top_comments(post)
                            }
                            data.append(post_data)
                            count = len(data)

                            # Print progress every 10 posts
                            if len(data) % 10 == 0:
                                #print(f"Fetched {count} relevant posts for keyword '{keyword}'")
                                logging.info(f"Fetched {count} relevant posts for keyword '{keyword}'")
                            time.sleep(0.1)

                # Save filtered data incrementally to CSV
                if data:
                    df = pd.DataFrame(data)
                    df.drop_duplicates(subset=['title', 'content'], inplace=True)
                    save_to_csv(df, company)
                    logging.info(f"{company}: Completed fetching for keyword '{keyword}' with {len(data)} posts saved.")
                    data.clear()  # Clear data after saving to CSV

                break  # Exit retry loop if successful

            except (praw.exceptions.APIException, 
                    prawcore.exceptions.RequestException, 
                    ConnectionError, 
                    Exception) as e:
                retry_attempts += 1
                wait_time = max(60, min(90 * (2 ** retry_attempts), 1200))
                
                if isinstance(e, praw.exceptions.APIException):
                    logging.warning(f"API rate limit hit. Attempt {retry_attempts}/5. Waiting {wait_time/60:.1f} minutes.")
                else:
                    logging.warning(f"Error: {type(e).__name__}. Attempt {retry_attempts}/5. Waiting {wait_time/60:.1f} minutes.")
                
                time.sleep(wait_time + random.uniform(1, 5))
                
                if retry_attempts >= 5:
                    logging.error(f"Max retries exceeded for '{keyword}': {e}")
                    break

        time.sleep(0.2)  # Delay between keywords

    #print(f"Completed fetching posts for all keywords related to {company}.")
    logging.info(f"Completed fetching posts for all keywords related to {company}.")

# Function to collect data for all companies and their keywords
def collect_reddit_data(company_keywords, limit=limit):
    #print("Starting data collection for top-scoring posts in keyword searches...")
    logging.info("Starting data collection for top-scoring posts in keyword searches...")
    for company, keywords in company_keywords.items():
        #print(f"Starting data collection for company: {company}")
        logging.info(f"Starting data collection for company: {company}")
        fetch_keyword_data(company, keywords, limit)
    #print("Data collection completed.")
    logging.info("Data collection completed.")

# Main execution
if __name__ == "__main__":
    # Start data collection for each company and save results in separate CSV files
    collect_reddit_data(company_keywords, limit=limit)
