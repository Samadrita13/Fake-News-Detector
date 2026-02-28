import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0   # 0 = FAKE
true["label"] = 1   # 1 = REAL

# Keep only text column
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Remove missing or empty text
data = data.dropna(subset=["text"])
data = data[data["text"].str.strip() != ""]

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
data.to_csv("data/news.csv", index=False)

print("news.csv created successfully!")
print("Dataset shape:", data.shape)
print(data["label"].value_counts())