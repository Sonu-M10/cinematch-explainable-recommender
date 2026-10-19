import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/movies_cleaned.csv")

print("Starting movies:", len(df))

# Require overview and genres
df = df.dropna(subset=["overview", "genres"]).copy()

print("After requiring overview and genres:", len(df))

# Keep movies with at least 50 votes
df = df[df["vote_count"] >= 50].copy()

print("After vote threshold:", len(df))

# Fill optional text fields
optional_columns = ["keywords", "directors", "cast"]

for column in optional_columns:
    df[column] = df[column].fillna("")

# Save the modeling dataset
df.to_csv("data/movies_model.csv", index=False)

print("\nFinal modeling dataset:", df.shape)

print("\nMovies by release year (2020 onward):")
df["release_date"] = pd.to_datetime(df["release_date"])

recent = df[df["release_date"].dt.year >= 2020]

print(
    recent["release_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nSample movies:")
print(
    df[["title", "release_date", "vote_average", "vote_count"]]
    .head(10)
    .to_string(index=False)
)