import pandas as pd
import math
from collections import Counter

class NewsData:
    def __init__(self, df: pd.DataFrame):
        """Handles news article dataframes."""
        self.df = df
        self.avg_doc_length = sum(len(row.split()) for row in self.df["Article Body"]) / len(self.df)

    def compute_tf(self, text):
        """Calculates term frequency for each word in a document."""
        words = text.split()
        word_count = len(words)
        tf = Counter(words)
        for word in tf:
            tf[word] /= word_count
        return tf

    def compute_idf(self):
        """Calculates inverse document frequency for each word in the corpus."""
        num_docs = len(self.df)
        idf = {}
        
        # Count how many documents contain each word
        for _, row in self.df.iterrows():
            words = set(row["Article Body"].split())
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        
        # Calculate IDF for each word
        for word, doc_count in idf.items():
            idf[word] = math.log(num_docs / doc_count)
        
        return idf

    def compute_tf_bm25(self, text, k1=1.5, b=0.75):
        """Calculates BM25-based term frequency for each word in a document."""
        words = text.split()
        word_count = len(words)
        doc_length = len(words)
        tf = Counter(words)
        
        for word in tf:
            tf[word] = (tf[word] * (k1 + 1)) / (tf[word] + k1 * (1 - b + b * (doc_length / self.avg_doc_length)))
        
        return tf

    def compute_tf_atf(self, text):
        """Calculates augmented term frequency for each word in a document."""
        words = text.split()
        max_tf = max(Counter(words).values())
        tf = Counter(words)
        
        for word in tf:
            tf[word] = 0.5 + 0.5 * (tf[word] / max_tf)
        
        return tf

    def tf_bm25(self):
        """Calculates the TF-BM25 score for each word in each document."""
        idf = self.compute_idf()
        tf_bm25_dict = {}

        for i, row in self.df.iterrows():
            tf_bm25 = self.compute_tf_bm25(row["Article Body"])
            tf_bm25_scores = {word: tf_bm25[word] * idf.get(word, 0) for word in tf_bm25}
            tf_bm25_dict[i] = tf_bm25_scores

        return tf_bm25_dict

    def tf_atf(self):
        """Calculates the TF-ATF score for each word in each document."""
        idf = self.compute_idf()
        tf_atf_dict = {}

        for i, row in self.df.iterrows():
            tf_atf = self.compute_tf_atf(row["Article Body"])
            tf_atf_scores = {word: tf_atf[word] * idf.get(word, 0) for word in tf_atf}
            tf_atf_dict[i] = tf_atf_scores

        return tf_atf_dict

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

# Calculate TF-BM25
tf_bm25_results = news_data.tf_bm25()
print("TF-BM25 Results:")
for article_idx, tf_bm25_scores in tf_bm25_results.items():
    print(f"Article {article_idx} TF-BM25 scores:")
    for word, score in tf_bm25_scores.items():
        print(f"  {word}: {score}")
    print("\n")

# Calculate TF-ATF
tf_atf_results = news_data.tf_atf()
print("TF-ATF Results:")
for article_idx, tf_atf_scores in tf_atf_results.items():
    print(f"Article {article_idx} TF-ATF scores:")
    for word, score in tf_atf_scores.items():
        print(f"  {word}: {score}")
    print("\n")
