import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
import time
import os

# Load the combined news data CSV file
combined_news_data = pd.read_csv('Seb_Folder/processed/combined_news_data.csv')

# Extract the unique source URLs from the data
url_sources = combined_news_data['url'].unique()
total_successes = 0

# Print all extracted unique URL sources
print(f"Total unique URL sources: {len(url_sources)}, Total successful URL pulls so far: 0")

# Start timer
start_time = time.time()

# Check if previous progress exists
if os.path.exists('scraped_main_texts_unique.csv'):
    scraped_data = pd.read_csv('scraped_main_texts_unique.csv')
    completed_urls = set(scraped_data['url'].tolist())
    main_texts = scraped_data.values.tolist()
else:
    completed_urls = set()
    main_texts = []

# Check if previous log exists
if os.path.exists('scraping_log.csv'):
    log_data = pd.read_csv('scraping_log.csv')
    log_entries = log_data.values.tolist()
    total_failures = log_data[log_data['status'] == 'Failed'].shape[0]
    total_successes = log_data[log_data['status'] == 'Success'].shape[0]
else:
    log_entries = []
    total_failures = 0
    total_successes = 0

# Scrape main text body from unique URLs and save periodically to a new CSV
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

print("\nScraping main text body from unique URLs...")
for url in url_sources:
    if url in completed_urls:
        continue

    attempt_start_time = time.time()
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            main_text = ' '.join([p.get_text() for p in paragraphs])
            main_texts.append((url, main_text))
            total_successes += 1
            log_entries.append((url, "Success", time.time() - attempt_start_time, 1))
        else:
            main_texts.append((url, "Failed to retrieve content"))
            log_entries.append((url, "Failed", time.time() - attempt_start_time, 1))
            total_failures += 1
    except requests.RequestException as e:
        main_texts.append((url, "Failed to retrieve content"))
        log_entries.append((url, "Failed", time.time() - attempt_start_time, 1))
        total_failures += 1

    # Save progress periodically every 100 iterations
    if len(main_texts) % 10 == 0:
        print(f"Scraped {len(main_texts)} URLs... Total successes: {total_successes}, Total failures: {total_failures}")
        with open('scraped_main_texts_unique.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['url', 'main_text'])
            writer.writerows(main_texts)

        with open('scraping_log.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['url', 'status', 'total_time', 'attempts'])
            writer.writerows(log_entries)

# Save the scraped main texts to a CSV file
with open('scraped_main_texts_unique.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['url', 'main_text'])
    writer.writerows(main_texts)

# Save the log of problematic URLs to a CSV file
with open('scraping_log.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['url', 'status', 'total_time', 'attempts'])
    writer.writerows(log_entries)

# End timer
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"\nScraping completed and saved to 'scraped_main_texts_unique.csv' in {time_taken:.2f} seconds")
print(f"Log of problematic URLs saved to 'scraping_log.csv'")
print(f"Total Failures: {total_failures}, Total Successes: {total_successes}")
