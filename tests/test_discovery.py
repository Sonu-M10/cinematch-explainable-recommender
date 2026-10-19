import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from discovery import apply_hard_filters, discover


class DiscoveryTests(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame([
            {
                "id": 1,
                "title": "One",
                "genres": "Comedy",
                "runtime": 90,
                "release_date": "2020-01-01",
                "original_language": "en",
                "vote_count": 100,
                "weighted_rating": 7.5,
            },
            {
                "id": 2,
                "title": "Two",
                "genres": "Horror",
                "runtime": 140,
                "release_date": "2018-01-01",
                "original_language": "fr",
                "vote_count": 200,
                "weighted_rating": 8.0,
            },
            {
                "id": 3,
                "title": "Three",
                "genres": "Comedy, Drama",
                "runtime": None,
                "release_date": "2022-01-01",
                "original_language": "en",
                "vote_count": 300,
                "weighted_rating": 9.0,
            },
        ])

    def test_runtime_filter(self):
        result = apply_hard_filters(
            self.df,
            max_runtime=120
        )
        self.assertEqual(result["id"].tolist(), [1])

    def test_missing_runtime_column(self):
        with self.assertRaises(ValueError):
            apply_hard_filters(
                self.df.drop(columns="runtime"),
                max_runtime=120
            )

    def test_excluded_genres(self):
        result = apply_hard_filters(
            self.df,
            excluded_genres=["Horror"]
        )
        self.assertEqual(set(result["id"]), {1, 3})

    def test_year_and_language(self):
        result = apply_hard_filters(
            self.df,
            language="en",
            year_min=2021
        )
        self.assertEqual(result["id"].tolist(), [3])

    def test_mood_and_hidden_movies(self):
        result = discover(
            self.df,
            mood="Funny",
            hidden_ids=[1]
        )
        self.assertEqual(int(result.iloc[0]["id"]), 3)
        self.assertNotIn(1, result["id"].tolist())

    def test_no_matching_movies(self):
        result = discover(
            self.df,
            genre="Western"
        )
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
