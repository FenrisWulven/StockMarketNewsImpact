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
from datasketch import MinHash, MinHashLSH, LeanMinHash
# Import stopwords from NLTK
from nltk.corpus import stopwords
from tqdm import tqdm
from random import seed
np.random.seed(42)
seed(42)

class NewsData:
    def __init__(self, df: pd.DataFrame):
        """Handles news article dataframes."""
        self.df = df  # Drop rows with missing body text
        self.df["cleaned"] = self.df["body"].apply(self.clean_text)  # Clean text column
        self.df["timestamp"] = self.df["timestamp"].apply(self.convert_date)  # Convert date column to datetime
        #self.vocabulary = self.build_vocabulary()
        #self.idf = self.compute_idf()

    def clean_text(self, text):
        """Standardizes and cleans text by lowercasing, removing punctuation, and extra whitespace."""
        try:
             text = text.lower()  # Lowercasing
             text = re.sub(r'(?:https?://|www\.)[^\s]+|(?:\b[a-zA-Z0-9.-]+\.(?:com|org|net|io)\b)', '', text)
             text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
             text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove special characters such as /, \, |, # etc.
             text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
        except:
            print(text)
            # print row number
            print(self.df[self.df['body'] == text].index)
            # print row
            print(self.df[self.df['body'] == text])

        # Remove stopwords and words "one", "time", "reddit"
        stop_words = set(stopwords.words('english')).union({"one", "reddit", "time", "zacks", "rank", "motley", "fool", "cookies",
                                                            "terms", "service", "privacy", "policy", "contact", "us", "advertising",
                                                            "about", "careers", "help", "site", "map", "copyright", "trademark",
                                                            "disclaimer", "accessibility", "preferences", "newsletter", "feedback",
                                                            "use", "site", "constitutes", "acceptance", "user", "agreement", "please",
                                                            "password", "forgot", "username", "email", "email", "password", "username",
                                                            "dont", "know", "company", "return", "stock", "market", "investment",
                                                            "herein", "represented", "said"})
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    
    def generate_minhash(self, doc, num_perm=256):
        """Generates a MinHash signature for a document."""
        minhash = MinHash(num_perm=num_perm)
        for word in set(doc.split()):  # Use set to avoid duplicate contributions
            minhash.update(word.encode('utf8'))
        return LeanMinHash(minhash)
    
    def delete_lsh(self):
        """Deletes the LSH index to free up memory."""
        del self.lsh
        del self.minhashes
    
    def build_lsh(self, threshold=0.2, num_perm=128):
        """Builds an LSH index for approximate similarity detection."""
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = {}  # Store MinHash objects for later use

        # TQDM this
        for idx, doc in enumerate(tqdm(self.df["cleaned"], desc="Building LSH index")):
            minhash = self.generate_minhash(doc, num_perm=num_perm)
            self.lsh.insert(idx, minhash)  # Insert into LSH
            self.minhashes[idx] = minhash  # Cache for querying
    
    def find_duplicates(self):
        """Finds clusters of near-duplicate documents."""
        duplicate_groups = []
        visited = set()

        for idx in self.minhashes:
            if idx in visited:
                continue
            group = self.lsh.query(self.minhashes[idx])  # Find similar documents
            duplicate_groups.append(group)
            visited.update(group)

        return duplicate_groups
    
    def merge_duplicates(self, duplicate_groups):
        """Merges near-duplicate documents into single nodes."""
        # Create a mapping from original document indices to merged node indices
        merged_mapping = {}
        keep_indices = set()
        remove_indices = set()
        
        for node_idx, group in enumerate(duplicate_groups):
            # Map each document index in the group to the representative node index
            merged_mapping.update({doc_idx: node_idx for doc_idx in group})
            
            # Keep the first document in the group and mark the rest for removal
            keep_indices.add(group[0])
            remove_indices.update(group[1:])

        # Remove duplicates from the DataFrame, keeping only representative rows
        self.df = self.df.drop(index=list(remove_indices)).reset_index(drop=True)
        
        # Update merged_mapping to account for new DataFrame indices
        updated_mapping = {}
        for old_idx, new_idx in enumerate(self.df.index):
            if old_idx in merged_mapping:
                updated_mapping[new_idx] = merged_mapping[old_idx]
        
        self.merged_mapping = updated_mapping

    def build_vocabulary(self):
        """Creates a consistent vocabulary across documents."""
        vocabulary = set()
        for article in self.df["cleaned"]:
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
        
        for article in self.df["cleaned"]:
            doc_count.update(set(article.split()))
        
        return {word: math.log((num_docs + 1) / (count + 1)) + 1 for word, count in doc_count.items()}

    def generate_vector(self, tf_dict):
        """Generates a sparse vector for a document based on TF and IDF."""
        vector = np.array([tf_dict.get(word, 0) * self.idf.get(word, 0) for word in self.vocabulary])
        return csr_matrix(vector)  # Convert to sparse matrix to save memory

    def compute_similarity_matrix(self, method="tfidf"):
        """Calculates pairwise cosine similarity for all document vectors."""
        # Ensure all vectors are created with the same vocabulary length
        vectors = [self.generate_vector(self.compute_tf(doc)).toarray()[0] for doc in self.df["cleaned"]]
        
        # Calculate cosine similarity
        return cosine_similarity(vectors)
    
    def compute_similarity_lsh(self):
        """Uses LSH to find similar document pairs."""
        for idx in self.minhashes:
            similar_docs = self.lsh.query(self.minhashes[idx])
            for sim_idx in similar_docs:
                if idx < sim_idx:  # Avoid duplicate pairs
                    yield idx, sim_idx
    
    def compute_similarity_incremental(self):
        """
        Calculates pairwise cosine similarity incrementally, yielding one pair at a time to save memory.
        """
        num_docs = len(self.df)

        for i in range(num_docs):
            tf_i = self.compute_tf(self.df["cleaned"].iloc[i])
            vec_i = self.generate_vector(tf_i)

            for j in range(i + 1, num_docs):
                tf_j = self.compute_tf(self.df["cleaned"].iloc[j])
                vec_j = self.generate_vector(tf_j)

                # Compute cosine similarity for the pair
                similarity = cosine_similarity(vec_i, vec_j)[0, 0]
                yield i, j, similarity
    
    def convert_date(self, date_str):
        """Converts a date string to a datetime object."""

        return pd.to_datetime(date_str)

# Example Usage
"""
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
"""
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

"""
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
"""