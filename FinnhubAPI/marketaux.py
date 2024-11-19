import requests
import json
import requests
import json
import time

# Parameters
api_token = 'jvY5DqlTyacARjdScci70wqnLPynkDCkwEtJ5BTY'


symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'META', 'AMZN', 'NVDA']
year = '2024'
start_date = f'{year}-01-01'
end_date = f'{year}-12-31'

# print(start_date, end_date)
limit = 20  # Max articles per request (depends on your plan)
max_requests = 300  # Adjust based on your plan
request_count = 0
all_articles = []

# Iterate through companies and paginate results
for symbol in symbols:
    page = 1
    while request_count < max_requests:
        url = (
            f'https://api.marketaux.com/v1/news/all'
            f'?api_token={api_token}'
            f'&symbols={symbol}'
            f'&filter_entities=true'
            f'&min_match_score=40'
            f'&published_after={start_date}'
            f'&published_before={end_date}'
            f'&language=en'
            f'&group_similar=true'
            f'&limit={limit}'
            f'&page={page}'
        )
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            all_articles.extend(data['data'])
            if len(data['data']) < limit:  # No more articles for this symbol
                break
            page += 1  # Go to the next page
            request_count += 1
            time.sleep(0.25)  # Rate limit handling
        else:
            break

# Save results to a JSON file
with open(f'filtered_market_news_{year}.json', 'w') as f:
    json.dump(all_articles, f, indent=4)

print(f"Retrieved {len(all_articles)} articles.")