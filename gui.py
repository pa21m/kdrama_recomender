from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, Listbox, Scrollbar, END

from .core import RecommenderModel


def run_gui(dataset_path: str | Path = "data/sample_kdrama.csv") -> None:
    """
    Launch the Tkinter GUI.
    """
    try:
        model = RecommenderModel.from_csv(dataset_path)
    except Exception as e:
        messagebox.showerror(
            "Dataset Error",
            f"Could not load dataset at: {dataset_path}\n\n{e}"
        )
        return

    root = tk.Tk()
    root.title("K-Drama Recommender")
    root.geometry("720x520")

    label = tk.Label(root, text="Enter K-Drama Title, Genre, or Year:", font=("Arial", 12))
    label.pack(pady=10)

    entry = tk.Entry(root, width=55, font=("Arial", 11))
    entry.pack(pady=5)

    result_list = None  # set below

    def show_recommendations() -> None:
        q = entry.get().strip()
        if not q:
            messagebox.showwarning("Input Error", "Please enter a K-drama name, genre, or year.")
            return

        result_list.delete(0, END)
        try:
            mode, results = model.smart_recommend(q, top_k=10)
        except Exception as e:
            result_list.insert(END, str(e))
            return

        # Header line
        result_list.insert(END, f"Mode: {mode} | Showing top {len(results)} results")
        result_list.insert(END, "-" * 90)

        for _, row in results.iterrows():
            name = row["Name"]
            yr = int(row["Year of release"])
            genre = row["Genre"]
            rating = row["Rating"]
            result_list.insert(END, f"{name} ({yr}) | {genre} | Rating: {rating}")

    entry.bind("<Return>", lambda _event: show_recommendations())

    button = tk.Button(root, text="Recommend", command=show_recommendations, font=("Arial", 11), bg="lightblue")
    button.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(pady=10, fill="both", expand=True)

    scrollbar = Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    result_list = Listbox(frame, width=110, height=18, font=("Arial", 10), yscrollcommand=scrollbar.set)
    result_list.pack(side="left", fill="both", expand=True)

    scrollbar.config(command=result_list.yview)

    # Help hint
    hint = tk.Label(
        root,
        text="Tip: try a title (e.g., 'Move to Heaven'), a genre token (e.g., 'Drama'), or a year (e.g., 2021).",
        font=("Arial", 9)
    )
    hint.pack(pady=5)

    root.mainloop()
