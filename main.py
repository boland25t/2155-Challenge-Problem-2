# 2.156 Challenge Problem 1
# Troy Boland, Andrew Castro

# =============================================================================
# ==========================          IMPORTS          ========================
# =============================================================================

from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend, use non-interactive PDF/PNG renderer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tqdm import tqdm

# =============================================================================
# =========================        PREDICTIONS         ========================
# =============================================================================


# Load data
grids = load_grids()
ratings = np.load("datasets/scores.npy")

def predict():
    """
    Runs predictions for all advisors with a progress bar.
    Calls fit_plot_predict() once per advisor and collects predictions.
    """
    all_predictions = []  # store predictions for all advisors

    print("\n🔄 Running advisor predictions...\n")

    # tqdm bar for advisors
    for advisor in tqdm(range(4), desc="Advisors", ncols=80):
        predictions = fit_plot_predict(grids, ratings, advisor)
        all_predictions.append(predictions)

    print("\n✅ All advisor predictions complete.\n")
    return all_predictions


# Run the pipeline
all_predictions = predict()

# ---- LaTeX PDF Build Section ----
try:
    subprocess.run(["latexmk", "-pdf", "plots.tex"], check=True)
    print("✅ LaTeX PDF rebuilt successfully.")
except FileNotFoundError:
    print("⚠️ LaTeXmk not found. Skipping PDF build.")
except subprocess.CalledProcessError:
    print("❌ LaTeX build failed. Check plots.tex for errors.")