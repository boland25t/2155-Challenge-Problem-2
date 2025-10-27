# 2.156 Challenge Problem 1
# Troy Boland, Andrew Castro

# =============================================================================
# ==========================          IMPORTS          ========================
# =============================================================================

from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tqdm import tqdm




# =============================================================================
#  Dump Random Grids to PDF
# =============================================================================
#  Troy Boland — 2.156 Challenge Problem 1
#
#  Randomly selects a set of grids and exports them to a multi-page PDF.
#  Each page shows one grid image, optionally annotated with its rating
#  from a selected advisor.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils_public import load_grids, plot_grid_image
import os

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_SAMPLES = 10             # Number of random grids to include
ADVISOR = 2                # Optional: pick which advisor’s ratings to include
RATINGS_PATH = "datasets/scores.npy"  # Optional ratings file
OUTFILE = f"random_grids_advisor_{ADVISOR}.pdf"
SEED = 42

# -----------------------------------------------------------------------------
# Load grids and ratings
# -----------------------------------------------------------------------------
print("📦 Loading grids...")
grids = load_grids()
n_total = len(grids)
print(f"Loaded {n_total} grids.")

ratings = None
if os.path.exists(RATINGS_PATH):
    ratings = np.load(RATINGS_PATH)
    print("✅ Ratings file found.")

# -----------------------------------------------------------------------------
# Random sampling derp
# -----------------------------------------------------------------------------
np.random.seed(SEED)
sample_indices = np.random.choice(n_total, N_SAMPLES, replace=False)
print(f"🧩 Selected grid indices: {sample_indices}")

# -----------------------------------------------------------------------------
# Plot & Export to PDF
# -----------------------------------------------------------------------------
print(f"🖨️ Writing {N_SAMPLES} grids to {OUTFILE}...")

with PdfPages(OUTFILE) as pdf:
    for i, idx in enumerate(sample_indices):
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_grid_image(grids[idx], on_ax=ax)

        title = f"Grid #{idx}"
        if ratings is not None:
            rating_val = ratings[idx, ADVISOR]
            if not np.isnan(rating_val):
                title += f" | Rating: {rating_val:.3f}"
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"✅ Successfully created {OUTFILE}")


# =============================================================================
# =========================        TESTSPACE           ========================
# =============================================================================
'''




ratings = np.load("datasets/scores.npy")
grids = load_grids()
advisor = 2
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor])
import pandas as pd
plot_ratings_histogram(ratings)
df = pd.DataFrame({
    "Grid_Index": np.arange(len(ratings_subset)),
    f"Advisor_{advisor}_Rating": ratings_subset
})

# Replace NaNs with empty strings for cleaner viewing
df.to_csv(f"advisor_{advisor}_ratings.csv", index=False, na_rep="")

print(f"✅ Saved advisor {advisor}'s ratings to advisor_{advisor}_ratings.csv")
'''