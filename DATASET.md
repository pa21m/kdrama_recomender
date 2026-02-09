# Dataset Notes

The full dataset comes from Kaggle:

Top 250 Korean Dramas (KDrama Dataset)
https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset

## Why the full CSV is not committed

Many Kaggle datasets are shared under terms that may restrict redistribution.  
To keep the repo safe to share publicly, this project git-ignores `data/kdrama.csv` and commits only a small sample CSV for demonstration.

## How to add the full dataset locally

1. Download the dataset from Kaggle.
2. Place the CSV as: `data/kdrama.csv`
3. Run:
   - GUI: `python -m kdrama_recommender --data data/kdrama.csv`
   - CLI: `python -m kdrama_recommender.cli --data data/kdrama.csv "Move to Heaven"`
