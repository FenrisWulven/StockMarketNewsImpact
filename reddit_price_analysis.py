import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def analyze_reddit_sentiment_vs_price():
    """Analyze and visualize Reddit sentiment correlation with stock prices"""
    
    # Create stock symbol to company name mapping
    stock_mapping = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'NVDA': 'Nvidia',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'GOOGL': 'Google',
        'META': 'Meta'
    }
    
    # Load stock data
    stocks_close = pd.read_csv("StockData/Googlefinance_stocks - Close_values.csv", skiprows=1)
    stocks_close = stocks_close.iloc[:, [0,1,3,5,7,9,11,13]]
    column_names = ['Date', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META']
    stocks_close.columns = column_names
    
    # Convert price columns to float
    for col in stocks_close.columns[1:]:
        stocks_close[col] = stocks_close[col].str.replace(',', '.').astype(float)
    
    # Convert dates with specific format
    stocks_close['Date'] = pd.to_datetime(
        stocks_close['Date'].str.replace('.', ':'),
        format='%d/%m/%Y %H:%M:%S'
    )
    
    # Load Reddit sentiment data
    reddit_sentiment = pd.read_csv(r"C:\Users\jbhan\Desktop\StockMarketNewsImpact\MarketNews\data\final\Reddit_2021_to_2024_with_sentiment.csv")
    reddit_sentiment['timestamp'] = pd.to_datetime(reddit_sentiment['timestamp'])
    
    # Calculate daily sentiment for each stock
    daily_sentiment = reddit_sentiment.groupby(
        ['stock', reddit_sentiment['timestamp'].dt.date]
    )['sentiment_score'].mean().reset_index()
    daily_sentiment['timestamp'] = pd.to_datetime(daily_sentiment['timestamp'])
    
    # Filter sentiment data for 2024
    daily_sentiment = daily_sentiment[daily_sentiment['timestamp'].dt.year == 2024]
    
    print("Unique stock values in Reddit data:", daily_sentiment['stock'].unique())
    print("Date range:", daily_sentiment['timestamp'].min(), "to", daily_sentiment['timestamp'].max())
    
    # Create subplots with more space at the top
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Adjust spacing - further increase top margin and space between plots
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)  # Decreased top from 0.90 to 0.85
    fig.suptitle('Reddit Sentiment vs Price Change (2-day MA) - 2024', fontsize=16, y=0.88)  # Moved title up from 0.93 to 0.88
    
    # Calculate correlations
    correlations = {}
    
    for idx, stock in enumerate(stocks_close.columns[1:]):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        
        # Calculate price changes first
        stock_data = stocks_close[stocks_close['Date'].dt.year == 2024]
        daily_prices = stock_data[stock_data['Date'].dt.time == pd.Timestamp('16:00:00').time()][['Date', stock]]
        daily_prices['Date'] = daily_prices['Date'].dt.date
        daily_prices.set_index('Date', inplace=True)
        price_changes = daily_prices[stock].pct_change()
        
        # Get stock-specific sentiment data using company name
        company_name = stock_mapping[stock]
        stock_sentiment = daily_sentiment[daily_sentiment['stock'] == company_name].copy()
        print(f"\n{stock} ({company_name}) sentiment data points in 2024: {len(stock_sentiment)}")
        
        if len(stock_sentiment) > 0:
            sentiment_ma = stock_sentiment.set_index('timestamp')['sentiment_score'].rolling(window=2).mean()
            print(f"Sentiment MA range: {sentiment_ma.min():.3f} to {sentiment_ma.max():.3f}")
            
            # Plot with different y-axis scales
            sentiment_line = sentiment_ma.plot(ax=ax1, color='blue', alpha=0.8, label='Reddit Sentiment MA', linewidth=2)
            price_line = price_changes.plot(ax=ax2, color='red', alpha=0.6, label='Price Change %')
            
            # Set y-axis limits for sentiment with wider range
            ax1.set_ylim(-1.0, 1.0)
            
            ax1.set_title(f'{stock} - Reddit Sentiment vs Price Change')
            ax1.set_ylabel('Sentiment Score', color='blue')
            ax2.set_ylabel('Daily Price Change %', color='red')
            
            # Rotate x-axis labels
            ax1.tick_params(axis='x', rotation=45)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Calculate correlation
            sentiment_idx = sentiment_ma.index.date
            price_idx = pd.to_datetime(price_changes.index)
            common_dates = set(sentiment_idx).intersection(set(price_idx.date))
            
            if common_dates:
                sentiment_aligned = sentiment_ma[[d in common_dates for d in sentiment_idx]]
                price_aligned = price_changes[[d in common_dates for d in price_idx.date]]
                correlation = sentiment_aligned.corr(price_aligned)
                correlations[stock] = correlation
        else:
            print(f"No sentiment data found for {stock}")
            price_changes.plot(ax=ax2, color='red', alpha=0.6, label='Price Change %')
            ax1.set_title(f'{stock} - No Reddit Sentiment Data Available')
            ax2.set_ylabel('Daily Price Change %', color='red')
            ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Maintain the top margin after tight_layout
    
    # Save high-quality figure
    plt.savefig('Figures/reddit_price_sentiment_ma.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create LaTeX table for correlations
    latex_table = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\caption{Reddit Sentiment-Price Change Correlations (2024)}\n"
        "\\label{tab:reddit_sentiment_price_correlations}\n"
        "\\begin{tabular}{lr}\n"
        "\\hline\n"
        "Stock & Correlation \\\\\n"
        "\\hline\n"
    )
    
    # Add correlations to table
    for stock, corr in sorted(correlations.items()):
        latex_table += f"{stock} & {corr:.3f} \\\\\n"
    
    latex_table += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    
    # Save LaTeX table
    with open('Figures/reddit_price_sentiment_correlations.tex', 'w') as f:
        f.write(latex_table)
    
    # Print correlations
    print("\nReddit Sentiment-Price Correlations (2024):")
    for stock, corr in sorted(correlations.items()):
        print(f"{stock}: {corr:.3f}")

if __name__ == "__main__":
    analyze_reddit_sentiment_vs_price() 