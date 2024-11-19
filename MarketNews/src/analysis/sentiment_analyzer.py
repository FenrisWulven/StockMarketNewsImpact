import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import os

# Define base path
BASE_PATH = r'C:\Users\jbh\Desktop\CompTools\StockMarketNewsImpact\MarketNews'

def create_article_similarity_network():
    """Create and analyze article similarity network using TF-IDF"""
    
    # Update file path to use 'processed' directory instead of 'FetchedNews'
    input_file = os.path.join(BASE_PATH, 'data', 'processed', 'finnhub_scraped_articles.csv')
    
    # Check if file exists and print helpful message if it doesn't
    if not os.path.exists(input_file):
        print(f"Error: Could not find input file at {input_file}")
        print("Please make sure the data file exists in the correct location.")
        return None, None
    
    # Load the scraped articles
    df = pd.read_csv(input_file)
    
    # Filter for successfully scraped articles
    successful_articles = df[
        (df['article_content'].str.len() > 100) & 
        (~df['article_content'].str.startswith('Error')) &
        (~df['article_content'].str.startswith('Content extraction failed'))
    ].copy()
    
    if len(successful_articles) < 2:
        print("Not enough successfully scraped articles to create similarity graph")
        return
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Create TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(successful_articles['article_content'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with metadata
    for idx, row in successful_articles.iterrows():
        G.add_node(idx, 
                  title=row.get('headline', row.get('title', 'No Title')),
                  symbol=row['symbol'],
                  date=row['datetime'])
    
    # Add edges for similar articles (threshold > 0.2)
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > 0.2:
                G.add_edge(successful_articles.index[i], 
                          successful_articles.index[j], 
                          weight=similarity_matrix[i, j])
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Set up the layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes colored by company
    node_colors = [plt.cm.tab10(hash(G.nodes[node]['symbol']) % 10) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.6)
    
    # Draw edges with width based on similarity
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
    
    # Add labels
    labels = {node: f"{G.nodes[node]['symbol']}\n{G.nodes[node]['date'][:10]}" 
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add title and legend
    plt.title("Article Similarity Network\n(Edges show content similarity, colors represent companies)", 
              pad=20, size=14)
    
    # Add company legend
    companies = list(set(successful_articles['symbol']))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(hash(company) % 10),
                                 label=company, markersize=10)
                      for company in companies]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Update output path for the plot
    output_dir = os.path.join(BASE_PATH, 'data', 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with updated path
    output_file = os.path.join(output_dir, 'article_similarity_network.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"\nNetwork visualization saved to: {output_file}")
    
    # Print analysis summary
    print("\nSimilarity Network Analysis:")
    print(f"Total articles analyzed: {len(successful_articles)}")
    print(f"Number of edges (similarities > 0.2): {G.number_of_edges()}")
    print(f"Average similarity score: {np.mean([d['weight'] for (u, v, d) in G.edges(data=True)]):.3f}")
    
    # Print most similar article pairs
    print("\nMost similar article pairs:")
    edge_data = [(u, v, d['weight']) for (u, v, d) in G.edges(data=True)]
    edge_data.sort(key=lambda x: x[2], reverse=True)
    
    for u, v, weight in edge_data[:5]:
        print(f"\nSimilarity: {weight:.3f}")
        print(f"Article 1: {G.nodes[u]['symbol']} - {G.nodes[u]['title']}")
        print(f"Article 2: {G.nodes[v]['symbol']} - {G.nodes[v]['title']}")
    
    return G, successful_articles

if __name__ == "__main__":
    G, articles_df = create_article_similarity_network()
