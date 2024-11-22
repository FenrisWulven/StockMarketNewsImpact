import pandas as pd
import math
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import networkx as nx
import community as community_louvain  # Install python-louvain package

class NewsData:
    def __init__(self, df: pd.DataFrame):
        """Handles news article dataframes."""
        self.df = df
        self.df["Cleaned Article"] = self.df["Body"].apply(self.clean_text)  # Clean text column
        self.df["Date"] = self.df["Date"].apply(self.convert_date)  # Convert date column to datetime
        self.vocabulary = self.build_vocabulary()
        self.idf = self.compute_idf()

    def clean_text(self, text):
        """Standardizes and cleans text by lowercasing, removing punctuation, and extra whitespace."""
        text = text.lower()  # Lowercasing
        text = re.sub(r'(?:https?://|www\.)[^\s]+', '', text)  # Remove URLs
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove special characters such as /, \, |, # etc.
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        # If preview is present in a word without spaces around it, remove that word from the text


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
    
    def convert_date(self, date_str):
        """Converts a date string to a datetime object."""
        # Check the format of the date string and convert to datetime D/M/Y
        if "-" in date_str:
            return pd.to_datetime(date_str)
        elif "/" in date_str:
            return pd.to_datetime(date_str, format="%d/%m/%Y")
        else:
            raise ValueError("Unknown date format. Please provide a valid date string.")

# Example Usage
data = {
    "Category": ["Economy", "Politics", "Tech", "Health"],
    "Short Description": [
        "Stock markets react to the new policy changes.",
        "The new tax bill has been introduced in Congress.",
        "Innovative AI technology is shaping the future.",
        "Recent health studies show promising results."
    ],
    "Body": [
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

# Louvain Community Detection
def louvain_community_detection(graph, use_package=True):
    """Applies the Louvain algorithm for community detection on an existing graph.
    
    Args:
        graph: The input graph.
        use_package (bool): If True, use the python-louvain package; otherwise, implement manually.
    
    Returns:
        dict: A dictionary mapping each node to its community.
    """
    if use_package:
        # Apply Louvain algorithm to detect communities using the package
        partition = community_louvain.best_partition(graph, weight='weight')
    else:
        # Manual implementation of a simple Louvain-like community detection (naive version)
        # Note: This is a simplified approach and not equivalent to the full Louvain algorithm
        nodes = list(graph.nodes())
        partition = {node: node for node in nodes}  # Start with each node in its own community
        change = True
        while change:
            change = False
            for node in nodes:
                best_community = partition[node]
                max_gain = 0
                current_community = partition[node]
                neighbor_communities = Counter([partition[neighbor] for neighbor in graph.neighbors(node)])
                for community, count in neighbor_communities.items():
                    if community == current_community:
                        continue
                    gain = count  # Simplified gain calculation based on number of neighbors in the community
                    if gain > max_gain:
                        max_gain = gain
                        best_community = community
                if best_community != current_community:
                    partition[node] = best_community
                    change = True
    return partition

def calculate_modularity(graph, partitions):
    m = graph.size(weight="weight")  # Total weight of edges
    degrees = dict(graph.degree(weight="weight"))  # Degree of each node
    modularity_score = 0

    node_to_community = {node: idx for idx, partition in enumerate(partitions) for node in partition}

    # Compute modularity
    for u in graph.nodes():
        for v in graph.nodes():
            if u == v or not graph.has_edge(u, v):
                A_uv = 0  # No edge between u and v
            else:
                A_uv = graph[u][v].get("weight", 1)  # Edge weight
            k_u = degrees[u]
            k_v = degrees[v]
            expected_weight = (k_u * k_v) / (2 * m)
            if node_to_community[u] == node_to_community[v]:  # Same community
                modularity_score += (A_uv - expected_weight)

    modularity_score /= (2 * m)
    return modularity_score

# Create a graph from the similarity matrix
G = nx.Graph()
num_nodes = similarity_matrix.shape[0]

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = similarity_matrix[i, j]
        if weight > 0:  # Only add edges with positive weights
            G.add_edge(i, j, weight=weight)

# Perform community detection (manually or using the package)
partition = louvain_community_detection(G, use_package=False)
print("\nLouvain Community Detection Result:")
print(partition)

# Output the community of each article
df['Community'] = df.index.map(partition)
print("\nData with Community Assignments:")
print(df[['Category', 'Community']])
