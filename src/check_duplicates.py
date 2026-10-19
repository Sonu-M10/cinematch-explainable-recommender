import pandas as pd


# Load modeling dataset
df = pd.read_csv("data/movies_model.csv")

# Find all movies titled Whiplash
whiplash = df[
    df["title"].str.lower() == "whiplash"
]

print("Movies titled Whiplash:")
print(
    whiplash[
        [
            "id",
            "title",
            "release_date",
            "directors",
            "vote_count",
            "overview"
        ]
    ].to_string(index=False)
)


# Extract release year
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

# Find records sharing the same title and release year
duplicates = df[
    df.duplicated(
        subset=["title", "release_year"],
        keep=False
    )
].copy()

print("\nRecords sharing a title and release year:")
print(len(duplicates))

print("\nNumber of title-year groups:")
print(
    duplicates.groupby(
        ["title", "release_year"]
    ).ngroups
)

print("\nSample duplicate candidates:")
print(
    duplicates[
        [
            "id",
            "title",
            "release_year",
            "directors",
            "vote_count"
        ]
    ]
    .sort_values(["title", "release_year"])
    .head(30)
    .to_string(index=False)
)