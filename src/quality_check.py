import pandas as pd

df = pd.read_csv("data/movies_cleaned.csv")

print("Dataset shape:", df.shape)

# Missing descriptive information
features = ["overview", "genres", "keywords"]

print("\nMissing descriptive features:")
print(df[features].isna().sum())

# How many movies have both overview and genres?
usable = df[
    df["overview"].notna()
    & df["genres"].notna()
]

print("\nMovies with overview and genres:", len(usable))

# Explore vote counts
print("\nVote count statistics:")
print(df["vote_count"].describe())

# Count movies above different vote thresholds
for threshold in [0, 10, 25, 50, 100]:
    count = (usable["vote_count"] >= threshold).sum()
    print(f"Movies with at least {threshold} votes: {count}")