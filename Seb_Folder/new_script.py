import pandas as pd
import math
from collections import Counter

class NewsData:
    def __init__(self, df: pd.DataFrame):
        """Handles news article dataframes."""
        self.df = df

    def compute_tf(self, text):
        """Calculates term frequency for each word in a document."""
        words = text.split()
        word_count = len(words)
        tf = Counter(words)  # Count the occurrences of each word
        for word in tf:
            tf[word] /= word_count  # Normalize by the total number of words
        return tf

    def compute_idf(self):
        """Calculates inverse document frequency for each word in the corpus."""
        num_docs = len(self.df)
        idf = {}
        
        # Count how many documents contain each word
        for _, row in self.df.iterrows():
            words = set(row["Article Body"].split())  # Use a set to count each word once per document
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        
        # Calculate IDF for each word
        for word, doc_count in idf.items():
            idf[word] = math.log(num_docs / doc_count)
        
        return idf

    def tfidf(self):
        """Calculates the TF-IDF score for each word in each document."""
        idf = self.compute_idf()  # Get IDF values for the entire corpus
        tfidf_dict = {}

        for i, row in self.df.iterrows():
            tf = self.compute_tf(row["Article Body"])  # Get TF values for the document
            tfidf = {word: tf[word] * idf.get(word, 0) for word in tf}  # Calculate TF-IDF for each word
            tfidf_dict[i] = tfidf  # Store TF-IDF scores for the document

        return tfidf_dict

# Sample data
data = {
    "Category": ["Economy", "Politics", "Tech", "Health"],
    "Short Description": [
        "Stock markets react to the new policy changes.",
        "The new tax bill has been introduced in Congress.",
        "Innovative AI technology is shaping the future.",
        "Recent health studies show promising results."
    ],
    "Article Body": [
        "The stock market experienced significant changes after recent policy announcements affecting various sectors.",
        "The newly introduced tax bill has sparked debates in Congress and could have long-term impacts.",
        "Artificial Intelligence is evolving with applications in multiple industries including healthcare and finance.",
        "A new health study indicates that certain lifestyle changes could lead to improved well-being and longevity."
    ],
    "Date": ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"],
    "Link": [
        "https://news.example.com/economy1",
        "https://news.example.com/politics1",
        "https://news.example.com/tech1",
        "https://news.example.com/health1"
    ]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Create an instance of NewsData
news_data = NewsData(df)

# Calculate TF-IDF manually
tfidf_results = news_data.tfidf()

# Print out the TF-IDF results for each article
for article_idx, tfidf_scores in tfidf_results.items():
    print(f"Article {article_idx} TF-IDF scores:")
    for word, score in tfidf_scores.items():
        print(f"  {word}: {score}")
    print("\n")
