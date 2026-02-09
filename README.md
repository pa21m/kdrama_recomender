# K-Drama Recommender (TF‑IDF + Cosine Similarity)

A simple content-based recommender that suggests Korean dramas using **TF‑IDF** features built from each show's **Synopsis + Genre + Cast**, then ranks similar shows using **cosine similarity**.

Includes:
- Tkinter GUI app
- CLI mode (terminal)
- Sample dataset (15 rows) so the repo runs out-of-the-box
- Instructions to download the full Kaggle dataset (not committed)

# Portfolio Project Summary
Problem

As Korean media has risen in popularity, Korean dramas have increased in viewrship. Viewers often struggle to decide which K-drama to watch next, especially with hundreds of titles available.
There is a need for a simple system that can recommend similar dramas based on story themes, genre, cast, and overall style, rather than just popularity rankings.

Approach

I built a content-based recommendation system using Python and machine learning techniques:

- Combined key text features: Synopsis + Genre + Cast

- Applied text cleaning and preprocessing (lowercasing, stop-word removal, punctuation filtering)

- Converted text into numerical representations using TF-IDF vectorisation

- Calculated similarity between dramas using cosine similarity

- Returned the Top-K most similar dramas based on a user query (title, genre, or year)

To make the system usable in real scenarios, I also developed:

- A Tkinter GUI for non-technical users

- A command-line interface (CLI) for quick testing and automation

- A modular package structure suitable for GitHub and reuse

Results & Impact

Successfully generates relevant drama recommendations based on narrative and thematic similarity rather than simple ratings.

Demonstrates practical application of:

Natural Language Processing (TF-IDF)

Similarity search (cosine similarity)

Software packaging and user interface design

## 1) Project Structure

```
kdrama-recommender/
├─ src/
│  └─ kdrama_recommender/
│     ├─ core.py        # model + recommenders
│     ├─ gui.py         # Tkinter GUI
│     ├─ cli.py         # command-line interface
│     └─ __main__.py    # python -m kdrama_recommender
├─ data/
│  ├─ sample_kdrama.csv # small sample committed for demo
│  └─ (kdrama.csv)      # full dataset (download yourself; gitignored)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## 2) Dataset

Source (Kaggle): **Top 250 Korean Dramas (KDrama Dataset)**  
- https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset

The full dataset is **not included** in the repo by default (to avoid licensing/redistribution issues).
Download it and place it here:

```
data/kdrama.csv
```

The code expects these columns (present in the Kaggle dataset):
`Name, Synopsis, Cast, Year of release, Genre, Rating`

## 3) Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

> **Linux note:** if Tkinter is missing, install `python3-tk` from your package manager.

## 4) Run the GUI

Using the sample dataset:

```bash
python -m kdrama_recommender
```

Using the full dataset you downloaded:

```bash
python -m kdrama_recommender --data data/kdrama.csv
```

(If you prefer, run the module directly:)
```bash
python -c "from kdrama_recommender.gui import run_gui; run_gui('data/kdrama.csv')"
```

## 5) Run in the Terminal (CLI)

```bash
python -m kdrama_recommender.cli "Move to Heaven"
python -m kdrama_recommender.cli "Drama"
python -m kdrama_recommender.cli 2021
```

Add `--data` for the full dataset:

```bash
python -m kdrama_recommender.cli --data data/kdrama.csv "Flower of Evil"
```

## 6) How it works (quick)

1. **Combine text fields**: `Synopsis + Genre + Cast`  
2. **Clean text**: lowercase, remove punctuation/digits/stopwords  
3. **TF‑IDF Vectorize** to numeric features  
4. **Cosine Similarity** between all dramas  
5. Return the top‑K most similar shows (or filter by genre/year)

## 7) Credits

- Dataset
Top 250 Korean Dramas (KDrama Dataset) – Kaggle
https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset

- Tutorial inspiration
Shaw, A. – Python Project: Building a Movie Recommendation System
https://medium.com/@abhishekshaw020/python-project-building-a-movie-recommendation-system-175f5f32aa10

This project adapts the core idea of content-based recommendation using TF-IDF and cosine similarity, and extends it with a modular package structure, GUI interface, and CLI usage.

Author
Pamela Mercado – MSc Data Science & Artificial Intelligence
Independently implemented and expanded as part of a machine learning portfolio project.
