import pandas as pd
import os

# List of dataset filenames
datasets = [
    "reddit_stock_data_Apple.csv", "reddit_stock_data_Microsoft.csv", "reddit_stock_data_Nvidia.csv" 
    # ,"reddit_stock_data_Tesla.csv", "reddit_stock_data_Amazon.csv", "reddit_stock_data_Google.csv", "reddit_stock_data_Meta.csv"
]

# Initialize an empty list to store DataFrames
all_dataframes = []

# Process each dataset
for dataset in datasets:
    df = pd.read_csv(dataset, encoding='utf-8')
    
    # Extract stock name from filename
    stock_name = os.path.basename(dataset).split('_')[-1].split('.')[0]  # Get the stock name
    df['stock'] = stock_name  # Add stock column
    
    # Combine title and content
    df['title'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # Set top comments to body
    df['body'] = df['top_comments'].fillna('')
    
    # Add source column
    df['source'] = 'reddit'
    
    # Keep only the required columns
    df = df[['stock', 'timestamp', 'title', 'body', 'source']]
    
    # Append the processed DataFrame to the list
    all_dataframes.append(df)

# Concatenate all DataFrames into one big DataFrame
big_df = pd.concat(all_dataframes, ignore_index=True)

# Save the concatenated DataFrame to a new CSV
output_filename = "Reddit_2021_to_2024.csv"
big_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"Processed and saved: {output_filename}")


# 7 datasets - each in a csv
# "subreddit": post.subreddit.display_name,
# "title": post.title,
# "timestamp": datetime.datetime.fromtimestamp(post.created_utc),
# "content": post.selftext,
# "score": post.score,
# "num_comments": post.num_comments,
# "author": post.author.name if post.author else "N/A",
# "top_comments": get_top_comments(post)


# "stock"
# "timestamp"
# "title" (title + body/content of reddit post | newsarticle title +- summary)
# "body" (top comments | article summary + text)
# "source" (reddit | news agency)