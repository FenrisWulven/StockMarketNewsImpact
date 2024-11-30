import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import time

# Load the combined news data CSV file
data = pd.read_csv("Seb_Folder/processed/combined_news_data.csv")

# Start timer
start_time = time.time()

# Scrape main text body from URLs and save to a new CSV
print("\nScraping main text body from URLs...")
main_texts = []

# Identify the top 10 most common sources based on total occurrences
source_counts = data['source'].value_counts()
top_10_sources = source_counts.head(10).index.tolist()

# Retry scraping for the top 10 most common sources to improve success rates
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

for index, row in data.head(5000).iterrows():
    if row['source'] in top_10_sources:
        url = row['url']
        retries = 3
        success = False
        while retries > 0 and not success:
            try:
                response = requests.get(url, headers=headers, timeout=20)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    paragraphs = soup.find_all('p')
                    main_text = ' '.join([p.get_text() for p in paragraphs])
                    main_texts.append((index, main_text))
                    success = True
                else:
                    retries -= 1
            except requests.RequestException as e:
                retries -= 1

        if not success:
            main_texts.append((index, "Failed to retrieve content"))
    else:
        # For non-top-10 sources, proceed with normal scraping
        url = row['url']
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                main_text = ' '.join([p.get_text() for p in paragraphs])
                main_texts.append((index, main_text))
            else:
                main_texts.append((index, "Failed to retrieve content"))
        except requests.RequestException as e:
            main_texts.append((index, "Error: " + str(e)))

# Save the scraped main texts to a CSV file
with open('scraped_main_texts_2.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['row_number', 'main_text'])
    writer.writerows(main_texts)

# End timer
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"\nScraping completed and saved to 'scraped_main_texts.csv' in {time_taken:.2f} seconds")
