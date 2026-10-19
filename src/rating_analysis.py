import pandas as pd

df = pd.read_csv("data/movies_model.csv")

# Calculate the average rating
vote_average_mean = df["vote_average"].mean()

# Calculate the median vote count
vote_count_median = df["vote_count"].median()

# Calculate the 75th percentile of vote count
vote_count_75th_percentile = df["vote_count"].quantile(0.75)

# Print all three values
print(f"Average rating: {vote_average_mean:.2f}")
print(f"Median vote count: {vote_count_median:.0f}")
print(f"75th percentile of vote count: {vote_count_75th_percentile:.0f}")