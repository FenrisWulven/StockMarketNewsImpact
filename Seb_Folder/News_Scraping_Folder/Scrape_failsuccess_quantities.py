import pandas as pd
import matplotlib.pyplot as plt

# Load the scraped texts CSV file
scraped_data = pd.read_csv('scraped_main_texts_2.csv')

# Load the combined news data CSV file
combined_news_data = pd.read_csv('Seb_Folder/processed/combined_news_data.csv')

# Initialize dictionaries to store failure and success counts per source
failure_counts = {}
success_counts = {}

# Iterate through the rows of scraped_main_texts and determine failure or success
for index, row in scraped_data.iterrows():
    main_text = str(row['main_text']).strip()
    source = combined_news_data.loc[index, 'source'] if index < len(combined_news_data) else 'Unknown'

    # Check if the row is a failure
    if main_text.startswith('Error:') or main_text.startswith('Failed to retrieve content'):
        if source in failure_counts:
            failure_counts[source] += 1
        else:
            failure_counts[source] = 1
    else:
        if source in success_counts:
            success_counts[source] += 1
        else:
            success_counts[source] = 1

# Calculate the total counts (failures + successes) for each source
total_counts = {}
for source in set(failure_counts.keys()).union(success_counts.keys()):
    total_counts[source] = failure_counts.get(source, 0) + success_counts.get(source, 0)

# Calculate the failure rate for each source
failure_rate = {source: (failure_counts.get(source, 0) / total_counts[source]) for source in total_counts}

# Sort sources by the total number of occurrences (failures + successes)
sorted_sources = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True))

# Get total number of failures and successes
total_failures = sum(failure_counts.values())
total_successes = sum(success_counts.values())

# Calculate average number of characters in successful texts, handling NaN values
successful_texts = scraped_data[~scraped_data['main_text'].str.contains('Error:|Failed to retrieve content', na=False)].dropna(subset=['main_text'])
average_text_length = successful_texts['main_text'].apply(lambda x: len(str(x))).mean()

# Get the top 5 sources with the most failures
top_5_failures = dict(sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)[:10])

# Prepare data for plotting
sources = list(top_5_failures.keys())
failures = [failure_counts[source] for source in sources]
successes = [success_counts.get(source, 0) for source in sources]
failure_percentages = [failure_rate[source] * 100 for source in sources]

# Create a bar chart for the top 5 sources with the most failures
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(sources))

bar1 = ax.bar(index, failures, bar_width, label='Failures', color='r')
bar2 = ax.bar([i + bar_width for i in index], successes, bar_width, label='Successes', color='g')

# Add percentage labels above bars
for i in range(len(sources)):
    ax.text(i, failures[i] + 1, f'{failure_percentages[i]:.2f}%', ha='center', color='black')

# Set chart details
ax.set_xlabel('Source')
ax.set_ylabel('Count')
ax.set_title('Top 5 Sources with the Most Failures')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(sources, rotation=45, ha='right')
ax.legend()

# Display the average text length
plt.figtext(0.15, 0.02, f'Average Text Length: {average_text_length:.2f} characters', fontsize=10, color='blue')

# Show the plot
plt.tight_layout()
plt.show()

# Print summary
print("\nTotal Summary:")
print(f"Total Failures: {total_failures}")
print(f"Total Successes: {total_successes}")
print(f"Average Text Length (successful texts): {average_text_length:.2f} characters")
