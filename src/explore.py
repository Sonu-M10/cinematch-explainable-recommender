import pandas as pd

# Load dataset
df = pd.read_csv("data/movies.csv")

print("Dataset shape:", df.shape)

# Convert release dates
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

# Release-year distribution
recent = df[
    df["release_date"].dt.year.between(2020, 2026)
]

print("\nMovies by release year:")
print(
    recent["release_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

# Missing values
important_columns = [
    "id",
    "title",
    "release_date",
    "overview",
    "genres",
    "keywords",
    "directors",
    "cast",
    "vote_average",
    "vote_count",
    "poster_path"
]

print("\nMissing values:")
print(df[important_columns].isna().sum())

# Duplicate analysis
print("\nDuplicate ID occurrences:")
print(df["id"].duplicated().sum())

print("\nExact duplicate rows:")
print(df.duplicated().sum())

# Remove exact duplicates in a temporary DataFrame
df_unique = df.drop_duplicates()

print("\nRemaining duplicate ID occurrences:")
print(df_unique["id"].duplicated().sum())

# Inspect remaining conflicting IDs
conflicting = df_unique[
    df_unique.duplicated(subset=["id"], keep=False)
]

print("\nSample records with conflicting IDs:")
print(
    conflicting[
        ["id", "title", "release_date", "vote_count"]
    ].sort_values("id").head(20).to_string(index=False)
)

# Release status
print("\nRelease status distribution:")
print(df["status"].value_counts())
# Investigate records sharing the same movie ID
movie_id = 2979

movie_records = df[df["id"] == movie_id]

print("\nInvestigating movie ID:", movie_id)
print("Number of records:", len(movie_records))

# Identify columns containing different values
different_columns = []

for column in df.columns:
    if movie_records[column].nunique(dropna=False) > 1:
        different_columns.append(column)

print("\nColumns with different values:")
print(different_columns)

print("\nValues in those columns:")
if different_columns:
    print(movie_records[different_columns].to_string(index=False))
else:
    print("No differences found.")

# Analyze all duplicated movie IDs
duplicate_groups = df[
    df.duplicated(subset=["id"], keep=False)
].groupby("id")

# Check for conflicting movie titles
conflicting_titles = 0

for movie_id, group in duplicate_groups:
    if group["title"].nunique(dropna=False) > 1:
        conflicting_titles += 1

print("\nMovie IDs with conflicting titles:")
print(conflicting_titles)

# Count which columns differ across duplicate groups
difference_counts = {}

for column in df.columns:
    difference_counts[column] = 0

for movie_id, group in duplicate_groups:
    for column in df.columns:
        if group[column].nunique(dropna=False) > 1:
            difference_counts[column] += 1

print("\nColumns that differ across duplicate groups:")

for column, count in sorted(
    difference_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    if count > 0:
        print(f"{column}: {count}")