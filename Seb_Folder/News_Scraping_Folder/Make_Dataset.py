import pandas as pd

# Load the combined news data CSV file
combined_news_data = pd.read_csv('Seb_Folder/processed/combined_news_data.csv')

# Load the scraped texts CSV file containing the URL and Body
scraped_data = pd.read_csv('Seb_Folder/scraped_main_texts_unique.csv')

# Create a dictionary from the scraped data where keys are URLs and values are the main texts
url_to_body = dict(zip(scraped_data['url'], scraped_data['main_text']))

# Initialize a list to store the final rows for the new dataset
combined_rows = []

# Group by URL in the combined news data to create a stock list for each unique URL
grouped = combined_news_data.groupby('url')

# Initialize counters for progress tracking
total_urls = len(grouped)
processed_count = 0
failed_count = 0
progress_interval = 100  # Set progress logging interval

for url, group in grouped:
    processed_count += 1

    # Get the body from the scraped data
    body = url_to_body.get(url, "Failed to retrieve content")

    # Skip entries where the body could not be retrieved
    if isinstance(body, float) or body.startswith('Error:') or body == "Failed to retrieve content":
        failed_count += 1
        continue

    # Get the stock list, timestamp, title, and source for the group
    stocks = group['symbol'].tolist()
    timestamp = group['datetime'].iloc[0]
    title = group['title'].iloc[0]
    source = group['source'].iloc[0]

    # Append the combined data to the list
    combined_rows.append({
        'Stocks': stocks,
        'Timestamp': timestamp,
        'Title': title,
        'Body': body,
        'Source': source
    })

    # Log progress at specified intervals
    if processed_count % progress_interval == 0:
        print(f"Progress: {processed_count}/{total_urls} URLs processed. Failures so far: {failed_count}")

# Create a new DataFrame from the combined rows
combined_df = pd.DataFrame(combined_rows)

# Save the new DataFrame to a CSV file
combined_df.to_csv('combined_news_with_bodies.csv', index=False)

print("New dataset saved as 'combined_news_with_bodies.csv'")
print(f"Total URLs processed: {processed_count}, Total failures: {failed_count}")
