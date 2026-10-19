import pandas as pd

df = pd.read_csv("data/movies_model.csv")

movie = df[df["title"] == "Inception"].iloc[0]

features = [
    "title",
    "overview",
    "genres",
    "keywords",
    "directors",
    "cast"
]

for feature in features:
    print(f"\n{feature.upper()}:")
    print(movie[feature])