import pandas as pd
import re

# Load data
data = pd.read_csv("data/news.csv")

def clean_text(text):
    text = str(text)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove punctuation but keep spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

data["text"] = data["text"].apply(clean_text)

# Save cleaned data
data.to_csv("data/news_cleaned.csv", index=False)

print("Text cleaned and saved!")