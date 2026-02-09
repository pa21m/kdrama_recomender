from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re
import string


REQUIRED_COLUMNS = ["Name", "Synopsis", "Cast", "Year of release", "Genre", "Rating"]


def _clean_text(text: str) -> str:
    """
    Lightweight text cleaning for recommendation.
    - lowercases
    - removes bracketed text
    - strips punctuation
    - removes tokens that include digits
    - removes English stopwords
    """
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)  # remove bracketed fragments
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\w*\d\w*", "", text)  # remove words containing digits
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


@dataclass
class RecommenderModel:
    """Holds the data and similarity matrix needed for recommendations."""
    data: pd.DataFrame
    similarity: "pd.DataFrame"  # actually a numpy array, but keep typing simple
    name_to_index: pd.Series

    @staticmethod
    def from_csv(csv_path: Union[str, Path]) -> "RecommenderModel":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {csv_path}. "
                "See README for how to download the Kaggle dataset."
            )

        data = pd.read_csv(csv_path)

        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing}. Found columns: {list(data.columns)}"
            )

        data = data[REQUIRED_COLUMNS].dropna().copy()
        data["Year of release"] = data["Year of release"].astype(int)

        data["combined_features"] = (
            data["Synopsis"].astype(str) + " " +
            data["Genre"].astype(str) + " " +
            data["Cast"].astype(str)
        ).map(_clean_text)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(data["combined_features"])
        similarity = cosine_similarity(tfidf_matrix)

        name_to_index = pd.Series(data.index, index=data["Name"]).drop_duplicates()

        return RecommenderModel(data=data, similarity=similarity, name_to_index=name_to_index)

    def recommend_by_name(self, name: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend dramas similar to a given title (supports fuzzy matching).
        Returns a DataFrame of top_k recommendations.
        """
        if not name or not name.strip():
            raise ValueError("Name must be non-empty.")
        name = name.strip().lower()

        matches = difflib.get_close_matches(
            name,
            self.data["Name"].str.lower().tolist(),
            n=1,
            cutoff=0.6,
        )
        if not matches:
            raise KeyError("K-drama not found. Please check spelling.")

        matched_name = self.data[self.data["Name"].str.lower() == matches[0]].iloc[0]["Name"]
        idx = int(self.name_to_index[matched_name])

        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Exclude the item itself at rank 0
        top = scores[1: top_k + 1]
        rec_indices = [i for i, _ in top]

        out = self.data.iloc[rec_indices][["Name", "Genre", "Year of release", "Rating"]].copy()
        out.insert(0, "Matched title", matched_name)
        return out

    def recommend_by_genre(self, genre: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top_k dramas that match a genre substring, sorted by rating.
        """
        if not genre or not genre.strip():
            raise ValueError("Genre must be non-empty.")
        genre = genre.strip().lower()

        filtered = self.data[self.data["Genre"].str.lower().str.contains(genre, na=False)]
        if filtered.empty:
            raise KeyError("No K-dramas found for this genre.")

        out = filtered.sort_values(by="Rating", ascending=False).head(top_k)
        return out[["Name", "Genre", "Year of release", "Rating"]].copy()

    def recommend_by_year(self, year: Union[int, str], top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top_k dramas released in a given year, sorted by rating.
        """
        try:
            year_int = int(year)
        except Exception as e:
            raise ValueError("Year must be an integer (e.g., 2021).") from e

        filtered = self.data[self.data["Year of release"] == year_int]
        if filtered.empty:
            raise KeyError(f"No K-dramas found for year {year_int}.")

        out = filtered.sort_values(by="Rating", ascending=False).head(top_k)
        return out[["Name", "Genre", "Year of release", "Rating"]].copy()

    def smart_recommend(self, query: str, top_k: int = 10) -> Tuple[str, pd.DataFrame]:
        """
        Decide whether query is a year, a genre keyword, or a title; then recommend.

        Returns:
            (mode, results_df)
            mode ∈ {"year", "genre", "title"}
        """
        q = (query or "").strip()
        if not q:
            raise ValueError("Query must be non-empty.")

        if q.isdigit() and len(q) == 4:
            return "year", self.recommend_by_year(int(q), top_k=top_k)

        # Fast genre check: treat as genre if it exactly matches a known genre token
        genres = set(
            g.strip().lower()
            for g in self.data["Genre"].astype(str).str.split(",").explode().tolist()
        )
        if q.lower() in genres:
            return "genre", self.recommend_by_genre(q, top_k=top_k)

        return "title", self.recommend_by_name(q, top_k=top_k)
