import pandas as pd
import ast
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import time
import networkx as nx
from adjustText import adjust_text
import re
import string
import nltk
from nltk.corpus import stopwords
np.random.seed(42) 

nltk.download('stopwords')
    
stock_dict = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META"
}

### Gradient Calculation

def average_gradient(df, stock, date_time, t):
    """
    Calculate the average gradient for a given stock around a specified date and time.
    
    Parameters:
    - df (DataFrame): The dataframe containing Date, stock prices as columns.
    - stock (str): The stock symbol for which to calculate the gradient (e.g., 'AAPL').
    - date_time (str or datetime): The reference date and time as a string or datetime object.
    - t (int): Number of timesteps before and after the date_time to calculate the gradient.
    
    Returns:
    - float: The average gradient (price change per timestep).
    """
    # Ensure date_time is in datetime format
    date_time = pd.to_datetime(date_time)

    # Determine indices for `t` steps before and `t` steps after, even if `date_time` is not in the data
    before_indices = df.index[df['Date'] <= date_time].to_numpy()[-(t[0]+1):]  # Last `t` steps before or equal
    after_indices = df.index[df['Date'] > date_time].to_numpy()[:t[1]]      # First `t` steps after

    if len(before_indices) < t[0]+1 or len(after_indices) < t[1]:
        return None, None
    
    after_indices = np.sort(np.append(after_indices, before_indices[-1]))
    before_prices = df.loc[before_indices, stock].to_numpy()
    avg_gradient_before = np.gradient(before_prices).mean()

    after_prices = df.loc[after_indices, stock]
    avg_gradient_after = np.gradient(after_prices).mean()

    return avg_gradient_before, avg_gradient_after

def get_stock_gradient_change_reddit(stock_df, reddit_df_series, t=(2, 2)):
    stock_input = reddit_df_series['stock']
    if isinstance(stock_input, list):
        stock_input = stock_input[0]  # Take the first element if it's a list
    if stock_input not in stock_dict and stock_input not in stock_dict.values():
        print(f"ERROR in stock_input: {stock_input}")
        return None
    # if reddit_df_series['stock'][0] not in stock_dict and reddit_df_series['stock'][0] not in stock_dict.values():
    #     return None


    if stock_input in stock_dict.values():
        stock = stock_input  # It's already a stock symbol
    elif stock_input in stock_dict:
        stock = stock_dict[stock_input]  # Convert company name to stock symbol
    else:
        print(f"Stock symbol not found for {stock_input}.")
        return None  # Not a valid stock symbol or company name
    date_time = reddit_df_series['timestamp']

    average_gradient_before, average_gradient_after = average_gradient(stock_df, stock, date_time, t)
    if average_gradient_before is None or average_gradient_after is None:
        print(f"Stock data not available for {stock} around {date_time}.")
        return None
    return average_gradient_after - average_gradient_before

### PCY Algorithm

def generate_hash_buckets(transactions, num_buckets):
    """
    Generate hash buckets for pairs of items in the first pass.
    
    Parameters:
    - transactions: List of transactions where each transaction is a list of items
    - num_buckets: Number of hash buckets to use
    
    Returns:
    - Dictionary mapping bucket indices to counts
    """
    buckets = {}
    for transaction in transactions:
        # Generate all possible pairs in the transaction
        for i in range(len(transaction)):
            for j in range(i + 1, len(transaction)):
                # Simple hash function: sum of indices modulo num_buckets
                bucket = (hash(transaction[i]) + hash(transaction[j])) % num_buckets
                buckets[bucket] = buckets.get(bucket, 0) + 1
    return buckets

def get_frequent_items(transactions, min_support):
    """
    Get frequent individual items from transactions.
    
    Parameters:
    - transactions: List of transactions
    - min_support: Minimum support threshold (as count)
    
    Returns:
    - Set of frequent items
    """
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    return {item for item, count in item_counts.items() if count >= min_support}

def pcy_algorithm(cluster_data, min_support_pct=0.05, num_buckets=50):
    """
    Implement the PCY algorithm to find frequent pairs.
    
    Parameters:
    - cluster_data: DataFrame containing the cluster data
    - min_support_pct: Minimum support threshold as a percentage
    - num_buckets: Number of hash buckets to use
    
    Returns:
    - List of frequent pairs with their support counts
    """
    # Prepare transactions from cleaned text
    transactions = []
    for text in cluster_data['cleaned']:
        words = set(text.split())  # Using set to remove duplicates within transaction
        transaction = list(words)
        if len(transaction) > 0:  # Only add non-empty transactions
            transactions.append(transaction)
    
    num_transactions = len(transactions)
    min_support = int(min_support_pct * num_transactions)  # Convert percentage to count
    
    # First Pass: Count singles and hash pairs to buckets
    frequent_items = get_frequent_items(transactions, min_support)
    hash_buckets = generate_hash_buckets(transactions, num_buckets)
    
    # Create bitmap of frequent buckets
    frequent_buckets = {bucket for bucket, count in hash_buckets.items() 
                       if count >= min_support}
    
    # Second Pass: Count frequent pairs that hash to frequent buckets
    pair_counts = {}
    for transaction in transactions:
        # Consider only frequent items
        freq_items_in_trans = [item for item in transaction if item in frequent_items]
        
        # Count pairs that hash to frequent buckets
        for i in range(len(freq_items_in_trans)):
            for j in range(i + 1, len(freq_items_in_trans)):
                item1, item2 = freq_items_in_trans[i], freq_items_in_trans[j]
                bucket = (hash(item1) + hash(item2)) % num_buckets
                
                if bucket in frequent_buckets:
                    pair = tuple(sorted([item1, item2]))  # Ensure consistent ordering
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Filter pairs by minimum support
    frequent_pairs = [(pair, count) for pair, count in pair_counts.items() 
                     if count >= min_support]
    
    return frequent_pairs

### Association Rules

def generate_association_rules(pairs_df, transactions, min_support_pct, confidence_threshold, lift_threshold):
    """
    Generate association rules from frequent pairs.

    Parameters:
    - pairs_df: DataFrame with columns ['pair', 'count'] from PCY results.
    - transactions: List of transactions used in the PCY algorithm.
    - min_support_pct: Minimum support threshold as a percentage.
    - confidence_threshold: Minimum confidence threshold to generate a rule.
    - lift_threshold: Minimum lift threshold to generate a rule.

    Returns:
    - DataFrame containing rules with 'antecedents', 'consequents', 'support', 'confidence', and 'lift'.
    """
    # Prepare data
    num_transactions = len(transactions)
    item_support = {}

    # Calculate support for individual items
    for transaction in transactions:
        for item in transaction:
            item_support[item] = item_support.get(item, 0) + 1

    # Normalize item support
    for item in item_support:
        item_support[item] /= num_transactions

    # Generate rules
    rules = []
    for _, row in pairs_df.iterrows():
        #     pairs_df['pair'] = pairs_df['pair'].apply(ast.literal_eval)  # Convert string to tuple
        pair = row['pair']
        #pair = ast.literal_eval(row['pair'])  # Convert string representation of the pair to a tuple
        pair_count = row['count']
        support_pair = pair_count / num_transactions
        
        # Filter pairs by minimum support
        if support_pair < min_support_pct:
            continue  # Skip pairs that don't meet the support threshold

        for item in pair:
            antecedent = item
            consequent = [x for x in pair if x != item][0]  # Other item in the pair
            
            # Calculate confidence and lift
            confidence = support_pair / item_support[antecedent] if item_support[antecedent] > 0 else 0
            lift = confidence / item_support[consequent] if item_support[consequent] > 0 else 0

            # Filter rules based on confidence and lift thresholds
            if confidence >= confidence_threshold and lift >= lift_threshold:
                rules.append({
                    'antecedents': [antecedent],
                    'consequents': [consequent],
                    'support': support_pair,
                    'confidence': confidence,
                    'lift': lift
                })

    return pd.DataFrame(rules)

def visualize_association_rules(rules_df, cluster_id, max_rules=20):
    """
    Visualize association rules for a cluster using a directed graph.
    
    Parameters:
    - rules_df: DataFrame containing association rules with 'antecedents', 'consequents', 'lift', and 'confidence'.
    - cluster_id: The ID of the cluster being analyzed.
    - max_rules: The maximum number of rules to visualize.
    """
    if rules_df.empty:
        print(f"No rules to visualize for cluster {cluster_id}")
        return
    
    # Limit to top rules by lift
    top_rules = rules_df.nlargest(max_rules, 'lift')
    
    # Create a directed graph
    G = nx.DiGraph()
    
    for _, rule in top_rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        for a in antecedents:
            for c in consequents:
                G.add_edge(a, c, lift=rule['lift'], confidence=rule['confidence'])
    
    # Generate positions for nodes
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for reproducibility
    
    # Node colors based on type (antecedents vs. consequents)
    node_colors = []
    for node in G.nodes():
        if any(node in list(rule['antecedents']) for _, rule in top_rules.iterrows()):
            node_colors.append('lightblue')  # Antecedents
        else:
            node_colors.append('lightgreen')  # Consequents
    
    # Edge weights based on lift
    edge_weights = [G[u][v]['lift'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    scaled_weights = [w / max_weight * 5 for w in edge_weights]  # Scale for better visibility
    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=8,
        edge_color='gray',
        width=scaled_weights,
        arrowsize=15
    )
    
    # Add edge labels for lift and confidence
    edge_labels = {(u, v): f"Lift: {G[u][v]['lift']:.2f}\nConf: {G[u][v]['confidence']:.2f}"
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title(f"Association Rules - Cluster {cluster_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"pcy_plots/association_rules_cluster_{cluster_id}.png", dpi=300)
    plt.close()

#### Visualization functions
def visualize_pair_gradients(pairs_df, cluster_data, cluster_id, num_pairs=200, title="All Reddit Data"):
    """
    Visualize word pairs with corresponding price gradient changes.
    
    Args:
        pairs_df: Pre-computed DataFrame with columns ['pair', 'count']
        cluster_data: DataFrame containing the cluster data
        cluster_id: ID of the cluster being analyzed
    """
    # Take top pairs by count
    top_pairs_df = pairs_df.nlargest(num_pairs, "count").copy()
    
    # Pre-compute word sets for each text (only once)
    text_word_sets = {idx: set(text.split()) for idx, text in cluster_data['cleaned'].items()}
    
    # Calculate gradients for each pair
    gradients = []
    for _, row in top_pairs_df.iterrows():
        pair_set = set(row['pair'])
        matching_indices = [idx for idx, word_set in text_word_sets.items() if pair_set.issubset(word_set)]
        
        avg_gradient = (
            cluster_data.loc[matching_indices, 'stock_gradient_change'].mean() if matching_indices else 0
        )
        gradients.append(avg_gradient)
    
    top_pairs_df['avg_gradient'] = gradients
    top_pairs_df['abs_gradient'] = top_pairs_df['avg_gradient'].abs()

    print("\n avg grad \n", top_pairs_df['avg_gradient'].head())
    print("Abs \n", top_pairs_df['abs_gradient'].head())

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(data=top_pairs_df, x="count", y="avg_gradient", alpha=0.6)
    
    # Collect text annotations
    texts = []
    
    # Annotate top 10 most frequent pairs
    top_frequent = top_pairs_df.nlargest(10, "count")
    for _, row in top_frequent.iterrows():
        texts.append(
            plt.text(
                row['count'],
                row['avg_gradient'],
                f"{row['pair'][0]}-{row['pair'][1]}",
                fontsize=8,
                color='blue'
            )
        )
    
    # Annotate top 10 highest gradient pairs
    top_highest = top_pairs_df.nlargest(10, "avg_gradient")
    for _, row in top_highest.iterrows():
        texts.append(
            plt.text(
                row['count'],
                row['avg_gradient'],
                f"{row['pair'][0]}-{row['pair'][1]}",
                fontsize=8,
                color='green'
            )
        )
    
    # Annotate top 10 lowest gradient pairs
    top_lowest = top_pairs_df.nsmallest(10, "avg_gradient")
    for _, row in top_lowest.iterrows():
        texts.append(
            plt.text(
                row['count'],
                row['avg_gradient'],
                f"{row['pair'][0]}-{row['pair'][1]}",
                fontsize=8,
                color='red'
            )
        )

    # Adjust text positions to avoid overlaps
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        force_text=(0.8, 1.5)
    )

    plt.title(f"Price Gradient of Frequent Word Pairs for {title}")    
    plt.xlabel("Count (Frequency)")
    plt.ylabel("Average Price Gradient Change")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"pcy_plots/Cluster_{cluster_id}_scatter_{num_pairs}.png", dpi=300)
    plt.close()

    print(f"Saved scatter plot for Cluster {cluster_id} with top frequent and impactful pairs.")

    print("Top 10 highest gradient pairs:")
    print(top_highest[['pair', 'avg_gradient']])
    
    print("Top 10 lowest gradient pairs:")
    print(top_lowest[['pair', 'avg_gradient']])

def visualize_pair_network(pairs_df, cluster_id):
    """
    Create a network graph of frequent word pairs.
    
    Args:
        pairs_df: Pre-computed DataFrame with columns ['pair', 'count']
        cluster_id: ID of the cluster being analyzed
    """
    import networkx as nx
    
    # Take top 30 pairs for visualization
    top_pairs_df = pairs_df.nlargest(30, "count")
    
    G = nx.Graph()
    
    # Add edges with weights
    for _, row in top_pairs_df.iterrows():
        G.add_edge(row['pair'][0], row['pair'][1], weight=row['count'])
    
    plt.figure(figsize=(12, 8))
    
    # Optimize layout calculation
    if len(G.nodes) > 50:
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        pos = nx.spring_layout(G, k=1)
    
    # Draw network
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights)
    nx.draw(G, pos,
           with_labels=True,
           node_color="skyblue",
           edge_color=edge_weights,
           width=[w / max_weight * 3 for w in edge_weights],
           node_size=800,
           font_size=8)
    
    plt.title(f"Frequent Word Pair Network (Cluster {cluster_id})")
    #plt.tight_layout()
    plt.savefig(f"pcy_plots/Cluster_{cluster_id}_network.png", dpi=300)
    plt.close()

def visualize_wordcloud(cluster_data, cluster_id):
    """
    Generate and save a word cloud for a given cluster.
    """
    text = ' '.join(cluster_data['cleaned'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top Keywords - Cluster {cluster_id}")
    plt.savefig(f"pcy_plots/Cluster_{cluster_id}_wordcloud.png", dpi=300)
    plt.close()

def summarize_critical_findings(cluster_data, frequent_pairs, cluster_id):
    """
    Summarize the most critical findings from the PCY analysis for a cluster.
    
    Args:
    - cluster_data: DataFrame containing the cluster data
    - frequent_pairs: List of tuples [(pair, count), ...] returned by PCY
    - cluster_id: ID of the cluster being analyzed
    
    Returns:
    - Dictionary containing the critical findings
    """
    findings = {}
    
    # 1. Basic Statistics
    findings['cluster_size'] = len(cluster_data)
    findings['num_frequent_pairs'] = len(frequent_pairs)
    
    # 2. Most Frequent Word Pairs
    if frequent_pairs:
        pairs_df = pd.DataFrame(frequent_pairs, columns=['pair', 'count'])
        top_pairs = pairs_df.nlargest(5, 'count')
        findings['top_pairs'] = [(pair, count) for pair, count in zip(top_pairs['pair'], top_pairs['count'])]
        
        # 3. Price Impact Analysis
        price_impact = []
        for pair, count in top_pairs.values:
            # Filter rows where both words in the pair exist
            mask = cluster_data['cleaned'].apply(
                lambda text: all(word in text.split() for word in pair)
            )
            avg_gradient = cluster_data[mask]['stock_gradient_change'].mean()
            price_impact.append((pair, avg_gradient))
        
        findings['price_impact'] = price_impact
        
        # 4. Overall Cluster Sentiment
        avg_gradient = cluster_data['stock_gradient_change'].mean()
        findings['avg_price_impact'] = avg_gradient
    
    # Print summary
    print(f"\n=== Critical Findings for Cluster {cluster_id} ===")
    print(f"Cluster Size: {findings['cluster_size']}")
    print(f"Number of Frequent Pairs: {findings['num_frequent_pairs']}")
    
    if frequent_pairs:
        print("\nTop 5 Most Frequent Word Pairs:")
        for pair, count in findings['top_pairs']:
            print(f"  {pair[0]} & {pair[1]}: {count} occurrences")
        
        print("\nPrice Impact of Top Pairs:")
        for pair, impact in findings['price_impact']:
            impact_str = "positive" if impact > 0 else "negative"
            print(f"  {pair[0]} & {pair[1]}: {impact_str} ({impact:.4f})")
        
        print(f"\nOverall Cluster Price Impact: {findings['avg_price_impact']:.4f}")
    
    return findings

#### Clean Reddit Data
def clean_ad_content(text):
    motley_pattern = r'^founded in 1993, the motley fool is a financial services company dedicated to making the world smarter, happier, and richer\..*?learn more'
    zacks_pattern = r'^we use cookies to understand how you use our site and to improve your experience\..*?terms of service apply\.'
    cleaned_text = re.sub(motley_pattern, '', text, flags=re.DOTALL)
    if re.search(zacks_pattern, text, flags=re.DOTALL):
        return None  # Remove the entire row if it matches the Zacks pattern
    return re.sub(zacks_pattern, '', cleaned_text, flags=re.DOTALL)

def is_valid_body(text):
    return text is not None and len(text) >= 30

def clean_text(text):
    """Standardizes and cleans text by lowercasing, removing punctuation, and extra whitespace."""
    try:
            text = text.lower()  # Lowercasing
            text = re.sub(r'(?:https?://|www\.)[^\s]+|(?:\b[a-zA-Z0-9.-]+\.(?:com|org|net|io)\b)', '', text)
            text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove special characters such as /, \, |, # etc.
            text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    except:
        print("Error in cleaning text")
        # print(text)
        # # print row number
        # print(self.df[self.df['body'] == text].index)
        # # print row
        # print(self.df[self.df['body'] == text])
    # Remove stopwords and words "one", "time", "reddit"
    stop_words = set(stopwords.words('english')).union({"one", "reddit", "time", "zacks", "rank", "motley", "fool", "cookies",
        "terms", "service", "privacy", "policy", "contact", "us", "advertising",
        "about", "careers", "help", "site", "map", "copyright", "trademark",
        "disclaimer", "accessibility", "preferences", "newsletter", "feedback",
        "use", "site", "constitutes", "acceptance", "user", "agreement", "please",
        "password", "forgot", "username", "email", "email", "password", "username",
        "dont", "know", "company", "return", "stock", "market", "investment",
        "herein", "represented", "said", "anything", "even", "like", "people", "get", "would", "got", 
        "last", "went", "see", "look", "looked", "looking", "also", "could", "know", "knows", "knowing", "known",
        "deleted", "removed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "anything", "seen", "im", "pretty", "much", "since", "still", "thats", "thing", "things", "though", "thought",
        "isnt", "youre", "theyre", "dont", "doesnt", "didnt", "cant", "couldnt", "wont", "wouldnt", "shouldnt", "shoudlnt",
        "put", "day", "way", "think", "actually", "put", "still", "back", "go", "going", "gone", "went", "come", "coming", "came",
        "want", "many", "every", "really", "come", "feel", "feeling", "felt", "make", "makes", "made", "made", "made", "made",
        "friend", "asked", "make", "going", " want", "enough", "kind", "kinda", "kind", "kinda", "kind", "kinda", "kind", "kinda",
        "going", "really", "everything", "work", "need", "needmake", "say", "back", "family","human", "told",
        "anyone", "theres", "take", "place", "bot", "questions", "automatically", "action"
        "comment", "submission", "post", "give", "may", "everyone", "someone", "something", "nothing", "anything",
        "ive", "wanted", "around", "part", "without", "ask", "already", "use", "used", "using", "uses", "user", "users",
        "else", "wife", "husband", "son", "daughter", "mother", "father", "brother", "sister", "cousin", "aunt", "uncle",
        "one","two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",})
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


##### MAIN ### 
if __name__ == "__main__":
    # Paths
    PATH_COMMUNITIES = 'news_communities.csv'
    PATH_REDDIT = 'Reddit_2021_to_2024_with_sentiment.csv'
    PATH_STOCK_DATA = 'Stock_prices_2021_to_2024.csv'

    MIN_CLUSTER_SIZE = 50  # Minimum cluster size
    TIME = (2, 2) # Number of timesteps to look before and after the date_time to calculate the gradient.
    # a timestep is a row in the stock data which contains opening and closing price for each day. 
    # So (2,2) timesteps is looking at 1 day (2 prices) before and 1 day after the date_time.

    #### LOAD NEWS DATA ####
    df_communities = pd.read_csv(PATH_COMMUNITIES)
    print("Length of df_communities:", len(df_communities))
    # summary of df_communities
    #print("Summary of df_communities:\n", df_communities.describe())
    # Convert stock column from string representation of list to actual list
    df_communities['stock'] = df_communities['stock'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Remove any rows where community value is missing/NaN which means that the article was not assigned to a community
    df_communities.dropna(subset=['community'], inplace=True)
    print("After removing rows where community is NaN:", len(df_communities))
    # Convert community values to integers (handles cases where they may be floats)
    df_communities['community'] = df_communities['community'].apply(lambda x: int(x) if isinstance(x, float) else x)
    # Convert timestamp strings to datetime objects
    df_communities['timestamp'] = pd.to_datetime(df_communities['timestamp'])
    # Remove timezone info from timestamps to match stock data format
    df_communities['timestamp'] = df_communities['timestamp'].dt.tz_localize(None)

    #### LOAD REDDIT DATA ####
    # Combine title and body for sentiment analysis
    reddit_df = pd.read_csv(PATH_REDDIT)
    #print("Length of reddit_df:", len(reddit_df))
    # make sure timestamp is in datetime format
    reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
    reddit_df['timestamp'] = reddit_df['timestamp'].dt.tz_localize(None)
    # Rename 'source' column to 'Source'
    reddit_df = reddit_df.rename(columns={"source": "Source"})
    # Preprocess reddit_df['stock'] to match company_colors keys
    reddit_df['stock'] = reddit_df['stock'].replace(stock_dict)    
    # Combine title and body 
    #reddit_df['cleaned'] = reddit_df['title'] + ' ' + reddit_df['body'].fillna('')
    reddit_df['cleaned'] = reddit_df['full_text']
    # clean the text by converting to lowercase
    reddit_df['cleaned'] = reddit_df['cleaned'].astype(str).str.lower()
    # Apply the cleaning function-unwanted ad-related content (just in case it also is in the reddit data)
    original_clean = reddit_df['cleaned'].copy()
    reddit_df['cleaned'] = reddit_df['cleaned'].apply(clean_ad_content)
    # Remove stop words and reddit-specific words from bad spelling and online-linguistics
    reddit_df['cleaned'] = reddit_df['cleaned'].apply(lambda x: clean_text(x))
    # Drop rows where 'cleaned' is None or contains empty strings or very short content
    reddit_df = reddit_df[reddit_df['cleaned'].apply(is_valid_body)]
    # Re-index both original and cleaned data to align properly for comparison
    original_clean_aligned = original_clean.loc[reddit_df.index]
    # Convert it to a string just in case its a list
    reddit_df['stock'] = reddit_df['stock'].apply(lambda x: x if isinstance(x, str) else x)
    #print("\nType of reddit_df['stock']:", type(reddit_df['stock']))
    #print("Unique values in reddit_df['stock']:", reddit_df['stock'].unique())

    #### LOAD STOCK DATA ####
    df_stock = pd.read_csv(PATH_STOCK_DATA)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    # data time range of stock data
    print("\nData time range of stock data:", df_stock['Date'].min(), "to", df_stock['Date'].max())
    # data time range of df_communities
    print("Data time range of df_communities:", df_communities['timestamp'].min(), "to", df_communities['timestamp'].max())
    # data time range of reddit_df
    print("Data time range of reddit_df:", reddit_df['timestamp'].min(), "to", reddit_df['timestamp'].max())
    
    # plot histogram of dates for stock data overlayed with df_communities
    plt.hist(df_stock['Date'], bins=100, alpha=0.5, label='Stock Data')
    plt.hist(df_communities['timestamp'], bins=100, alpha=0.5, label='df_communities')
    plt.legend()
    plt.title("Dates Histogram")
    plt.xlabel("Date")
    plt.ylabel("Frequency")
    plt.savefig("dates_histogram.png")
    plt.close()

    #### PREPROCESS GRADIENT DATA ####
    # Remove rows outside of stock data range so we only keep data from the last 3 years
    df_communities = df_communities[df_communities['timestamp'] >= df_stock['Date'].min()]
    df_communities = df_communities[df_communities['timestamp'] <= df_stock['Date'].max()]
    print("After removing rows outside stock data range:",len(df_communities))
    reddit_df = reddit_df[reddit_df['timestamp'] >= df_stock['Date'].min()]
    reddit_df = reddit_df[reddit_df['timestamp'] <= df_stock['Date'].max()]
    print("After removing rows outside stock data range:",len(reddit_df))

    # Add stock gradient column to both news and reddit data
    df_communities['stock_gradient_change'] = df_communities.apply(lambda row: get_stock_gradient_change_reddit(df_stock, row, t=TIME), axis=1)
    df_communities.dropna(subset=['stock_gradient_change'], inplace=True)
    reddit_df['stock_gradient_change'] = reddit_df.apply(lambda row: get_stock_gradient_change_reddit(df_stock, row, t=TIME), axis=1)
    reddit_df.dropna(subset=['stock_gradient_change'], inplace=True)

    # histogram of stock_gradient_change
    plt.figure(figsize=(10, 5))
    plt.hist(df_communities['stock_gradient_change'], bins=100)
    plt.title("Stock Gradient Change Histogram")
    plt.xlabel("Stock Gradient Change")
    plt.ylabel("Frequency")
    plt.savefig("stock_gradient_change_histogram.png")
    plt.close()

    ##### CLUSTERS #####
    # Get the top 3 largest clusters
    cluster_sizes = df_communities['community'].value_counts()
    # histogram of cluster sizes
    plt.hist(cluster_sizes, bins=100)
    plt.title("Cluster Sizes Histogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Frequency")
    plt.savefig("cluster_sizes_histogram.png")
    plt.close()
    # Top 5 clusters
    top_5_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].head(5).index.tolist()
    print(f"\nAnalyzing top 5 clusters with indexes: {top_5_clusters}")
    print("Top 5 Cluster sizes:", {cluster: cluster_sizes[cluster] for cluster in top_5_clusters})
    largest_cluster = top_5_clusters[0]
    second_largest_cluster = top_5_clusters[1]

    ##### PCY PARAMS #####
    NUM_PAIRS = 0.4  # Percentage of pairs to visualize for gradient analysis
    MAX_RULES = 20  # Maximum number of association rules to visualize
    NUM_BUCKETS = 200  # Number of hash buckets for PCY
    MIN_SUPPORT = 0.05  # Minimum support threshold as a percentage
    CONFIDENCE = 0.7
    LIFT = 1.5

    ### DO PCY FOR ALL REDDIT DATA
    def run_pcy_and_plot(df, data_name, plot_title, clustering=False, cluster_id=None):

        print(f"\nRunning PCY algorithm for {data_name}...")
        
        if clustering:
            df = df[df['community'] == cluster_id]
        # get all transactions
        transactions = [list(set(text.split())) for text in df['cleaned']]

        # Run PCY algorithm - takes about 2mins
        start_time = time.time()
        frequent_pairs_all = pcy_algorithm(df, min_support_pct=MIN_SUPPORT, num_buckets=NUM_BUCKETS)
        pcy_time = time.time() - start_time
        print(f"PCY algorithm for all data: {pcy_time:.2f} seconds")

        #Analyze and visualize results
        if frequent_pairs_all:
            print(f"Found {len(frequent_pairs_all)} frequent pairs in the dataset.")
            # Convert to DataFrame for easier processing
            pairs_df = pd.DataFrame(frequent_pairs_all, columns=['pair', 'count'])
            pairs_df['support'] = pairs_df['count'] / len(df)

            # Save frequent pairs to CSV
            pairs_df.to_csv(f"{data_name}_pcy_pairs.csv", index=False)
            print(f"Saved all data PCY pairs to '{data_name}_pcy_pairs.csv'.")

            # Generate association rules
            rules_df = generate_association_rules(pairs_df, transactions, min_support_pct=MIN_SUPPORT,
                confidence_threshold=CONFIDENCE,lift_threshold=LIFT)
            if not rules_df.empty:
                visualize_association_rules(rules_df, cluster_id=data_name, max_rules=MAX_RULES)

            # Pair gradients visualization plotting with different percentages of all frequent pairs found
            percentages = [NUM_PAIRS, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
            for percentage in percentages:
                num_pairs = max(100, int(percentage * len(df)))
                print(f"Visualizing top {num_pairs} pair gradients for all data (percentage: {percentage*100:.0f}%)...")
                visualize_pair_gradients(pairs_df, df, cluster_id=data_name, num_pairs=num_pairs, title=plot_title)
            
            # Network visualization
            visualize_pair_network(pairs_df, cluster_id=data_name)

            # Wordcloud visualization
            visualize_wordcloud(df, cluster_id=data_name)

            # Generate association rules
            rules_df = generate_association_rules(pairs_df, transactions, min_support_pct=MIN_SUPPORT, confidence_threshold=CONFIDENCE, lift_threshold=LIFT)
            if not rules_df.empty:
                visualize_association_rules(rules_df, cluster_id=data_name, max_rules=MAX_RULES)

            print("Analysis and visualizations for all data completed.")
        else:
            print("No frequent pairs found for the entire dataset.")

    run_pcy_and_plot(reddit_df, data_name= "All_Reddit_Data", plot_title="All Reddit Data")
    run_pcy_and_plot(df_communities, data_name="All_News_Data", plot_title="All News Data")
    run_pcy_and_plot(df_communities, data_name="Largest_News_Cluster", plot_title="Largest News Cluster", clustering=True, cluster_id=largest_cluster)

    ### LOAD PCY RESULTS FROM CSV
    # transactions = [list(set(text.split())) for text in df_communities['cleaned']]
    # # load all_data_pcy_pairs.csv
    # pairs_df = pd.read_csv("all_data_pcy_pairs.csv")
    # pairs_df['pair'] = pairs_df['pair'].apply(ast.literal_eval)  # Convert string to tuple


    