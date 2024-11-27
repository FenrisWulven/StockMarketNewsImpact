import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

def chunk_text(text, max_length=512):
    """Split text into chunks of approximately max_length words"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_sentiment(text, tokenizer, model, device):
    """Get sentiment scores for a piece of text"""
    # Prepare text
    if pd.isna(text) or text.strip() == '':
        return 0, 0, 0, 0  # neutral for empty text
    
    # Combine longer texts into chunks
    chunks = chunk_text(text)
    chunk_scores = []
    
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            chunk_scores.append(scores.cpu().numpy())
    
    # Average scores across chunks
    if chunk_scores:
        avg_scores = np.mean(chunk_scores, axis=0)[0]
        # Calculate sentiment score as positive - negative
        sentiment_score = float(avg_scores[2] - avg_scores[0])  # positive - negative
        return sentiment_score, avg_scores[2], avg_scores[0], avg_scores[1]
    else:
        return 0, 0, 0, 0  # neutral for empty text

def main():
    # Load the model and tokenizer - using same model as news articles
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load Reddit data
    df = pd.read_csv('Reddit_2021_to_2024.csv')
    
    # Combine title and body for sentiment analysis
    df['full_text'] = df['title'] + ' ' + df['body'].fillna('')
    
    # Initialize lists for sentiment scores
    sentiment_scores = []
    sentiment_labels = []
    positive_scores = []
    negative_scores = []
    neutral_scores = []
    
    # Process each post
    for text in tqdm(df['full_text'], desc="Calculating sentiment"):
        score, pos, neg, neu = get_sentiment(text, tokenizer, model, device)
        
        # Determine sentiment label with thresholds
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        sentiment_scores.append(score)
        sentiment_labels.append(label)
        positive_scores.append(pos)
        negative_scores.append(neg)
        neutral_scores.append(neu)
    
    # Add sentiment columns to DataFrame
    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    df['positive_score'] = positive_scores
    df['negative_score'] = negative_scores
    df['neutral_score'] = neutral_scores
    
    # Save the results with _with_sentiment suffix
    output_file = 'Reddit_2021_to_2024_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\nSentiment Distribution:")
    print(df['sentiment_label'].value_counts(normalize=True).round(3) * 100, "%")
    
    print("\nAverage Sentiment Score by Stock:")
    print(df.groupby('stock')['sentiment_score'].mean().round(3))

if __name__ == "__main__":
    main()
