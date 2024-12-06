import pandas as pd
import math
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

class NewsData:
    def __init__(self, df: pd.DataFrame):
        """Handles news article dataframes."""
        self.df = df
        self.df["Cleaned Article"] = self.df["Article Body"].apply(self.clean_text)  # Clean text column
        self.vocabulary = self.build_vocabulary()
        self.idf = self.compute_idf()

    def clean_text(self, text):
        """Standardizes and cleans text by lowercasing, removing punctuation, and extra whitespace."""
        text = text.lower()  # Lowercasing
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        return text

    def build_vocabulary(self):
        """Creates a consistent vocabulary across documents."""
        vocabulary = set()
        for article in self.df["Cleaned Article"]:
            vocabulary.update(article.split())
        return vocabulary

    def compute_tf(self, text):
        """Calculates normalized term frequency using NumPy for efficiency."""
        words = text.split()
        word_count = len(words)
        tf = Counter(words)
        return {word: count / word_count for word, count in tf.items()}

    def compute_idf(self):
        """Calculates inverse document frequency for each word in the vocabulary."""
        num_docs = len(self.df)
        doc_count = Counter()
        
        for article in self.df["Cleaned Article"]:
            doc_count.update(set(article.split()))
        
        return {word: math.log((num_docs + 1) / (count + 1)) + 1 for word, count in doc_count.items()}

    def generate_vector(self, tf_dict):
        """Generates a sparse vector for a document based on TF and IDF."""
        vector = np.array([tf_dict.get(word, 0) * self.idf.get(word, 0) for word in self.vocabulary])
        return csr_matrix(vector)  # Convert to sparse matrix to save memory

    def compute_similarity_matrix(self, method="tfidf"):
        """Calculates pairwise cosine similarity for all document vectors."""
        # Ensure all vectors are created with the same vocabulary length
        vectors = [self.generate_vector(self.compute_tf(doc)).toarray()[0] for doc in self.df["Cleaned Article"]]
        
        # Calculate cosine similarity
        return cosine_similarity(vectors)


# Example Usage
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

# Create an instance of NewsData and calculate similarity matrix
news_data = NewsData(df)
similarity_matrix = news_data.compute_similarity_matrix()

# Print similarity matrix
print("TF-IDF Cosine Similarity Matrix:")
print(similarity_matrix)
