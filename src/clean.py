import pandas as pd

# 1. Load raw dataset
df = pd.read_csv("data/movies.csv")

print("Original shape:", df.shape)

# 2. Convert release dates
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

# 3. Keep released movies from 2010 onward
df = df[
    (df["status"] == "Released")
    & (df["release_date"].dt.year.between(2010, 2026))
].copy()

print("After date and status filtering:", df.shape)

# 4. Remove records without essential information
df = df.dropna(
    subset=["id", "title", "release_date"]
)

# 5. Calculate descriptive metadata completeness
metadata_columns = [
    "overview",
    "genres",
    "keywords",
    "directors",
    "cast",
    "poster_path"
]

# Treat empty strings as missing
for column in metadata_columns:
    df[column] = df[column].replace(r"^\s*$", pd.NA, regex=True)

df["metadata_score"] = (
    df[metadata_columns].notna().sum(axis=1)
)

# 6. Prefer the most complete record for each movie ID
df = df.sort_values(
    by=["metadata_score", "vote_count"],
    ascending=[False, False],
    kind="stable"
)

df = df.drop_duplicates(
    subset=["id"],
    keep="first"
)

# 7. Remove temporary scoring column
df = df.drop(columns=["metadata_score"])

# 8. Save cleaned dataset
df.to_csv("data/movies_cleaned.csv", index=False)

print("Final cleaned shape:", df.shape)
print("Remaining duplicate IDs:", df["id"].duplicated().sum())
print("Cleaned dataset saved successfully.")
