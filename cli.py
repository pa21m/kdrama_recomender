from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import RecommenderModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K-Drama recommender (title/genre/year).")
    p.add_argument("--data", default="data/sample_kdrama.csv", help="Path to the CSV dataset.")
    p.add_argument("--topk", type=int, default=10, help="Number of recommendations to show.")
    p.add_argument("query", nargs="?", help="Title, genre token, or year (e.g., 2021).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.query:
        print("Please provide a query. Example:\n  python -m kdrama_recommender.cli \"Move to Heaven\"")
        return 2

    model = RecommenderModel.from_csv(Path(args.data))
    mode, results = model.smart_recommend(args.query, top_k=args.topk)

    print(f"Mode: {mode}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
