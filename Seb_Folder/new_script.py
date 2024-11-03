import pandas as pd
import math
import numpy as np
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

    def generate_vector(self, tf_dict, idf):
        """Generates a vector from TF and IDF values for a document."""
        return np.array([tf_dict.get(word, 0) * idf.get(word, 0) for word in idf.keys()])

    def calculate_similarity(self, vectors, method="cosine"):
        """Calculates pairwise similarity between document vectors."""
        num_docs = len(vectors)
        similarity_matrix = np.zeros((num_docs, num_docs))

        for i in range(num_docs):
            for j in range(i, num_docs):
                if method == "cosine":
                    similarity = self.cosine_similarity(vectors[i], vectors[j])
                elif method == "dot_product":
                    similarity = np.dot(vectors[i], vectors[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symmetric matrix

        return similarity_matrix

    def cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

    def tfidf_vectors(self, method="tfidf"):
        """Generates document vectors for each TF-IDF variant."""
        idf = self.compute_idf()
        vectors = []

        for _, row in self.df.iterrows():
            if method == "tfidf":
                tf = self.compute_tf(row["Article Body"])
            elif method == "bm25":
                tf = self.compute_tf_bm25(row["Article Body"])
            elif method == "atf":
                tf = self.compute_tf_atf(row["Article Body"])

            vector = self.generate_vector(tf, idf)
            vectors.append(vector)

        return np.array(vectors)



# Sample data
# data = {
#     "Category": ["Economy", "Politics", "Tech", "Health"],
#     "Short Description": [
#         "Stock markets react to the new policy changes.",
#         "The new tax bill has been introduced in Congress.",
#         "Innovative AI technology is shaping the future.",
#         "Recent health studies show promising results."
#     ],
#     "Article Body": [
#         "The stock market experienced significant changes after recent policy announcements affecting various sectors.",
#         "The newly introduced tax bill has sparked debates in Congress and could have long-term impacts.",
#         "Artificial Intelligence is evolving with applications in multiple industries including healthcare and finance.",
#         "A new health study indicates that certain lifestyle changes could lead to improved well-being and longevity."
#     ],
#     "Date": ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"],
#     "Link": [
#         "https://news.example.com/economy1",
#         "https://news.example.com/politics1",
#         "https://news.example.com/tech1",
#         "https://news.example.com/health1"
#     ]
# }

# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)

# # Create an instance of NewsData
# news_data = NewsData(df)

# # Generate document vectors using TF-BM25
# bm25_vectors = news_data.tfidf_vectors(method="bm25")
# print("TF-BM25 Similarity Matrix:")
# bm25_similarity = news_data.calculate_similarity(bm25_vectors, method="dot_product")
# print(bm25_similarity)

# # Generate document vectors using TF-ATF
# atf_vectors = news_data.tfidf_vectors(method="atf")
# print("\nTF-ATF Cosine Similarity Matrix:")
# atf_similarity = news_data.calculate_similarity(atf_vectors, method="cosine")
# print(atf_similarity)
