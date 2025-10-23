# =============================================================================
#  Dump Grids From Striated Rating Ranges
# =============================================================================
#  Troy Boland — 2.156 Challenge Problem 1
#
#  Selects grids whose ratings fall within specific "striated" ranges
#  identified from histogram spikes, then exports 10 random samples per
#  range to a multi-page PDF for visual inspection.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils_public import load_grids, plot_grid_image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ADVISOR = 2                        # which advisor’s ratings to use
RATINGS_PATH = "datasets/scores.npy"
OUTFILE = f"striated_grids_advisor_{ADVISOR}.pdf"
SEED = 42
N_PER_RANGE = 10                   # number of grids per range

# Define the striated rating ranges (from your analysis)
RANGES = [
    (0.24, 0.25),   # spike 1
    (0.35, 0.36),   # spike 2
    (0.52, 0.54)    # spike 3 (the large one)
]

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("📦 Loading grids and ratings...")
grids = load_grids()
ratings = np.load(RATINGS_PATH)
ratings_subset = ratings[:, ADVISOR]

np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Select grids from each striated range
# -----------------------------------------------------------------------------
selections = []
for low, high in RANGES:
    mask = (ratings_subset >= low) & (ratings_subset < high)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        print(f"⚠️ No grids found in range {low:.2f}–{high:.2f}")
        continue
    chosen = np.random.choice(indices, min(N_PER_RANGE, len(indices)), replace=False)
    selections.append((low, high, chosen))
    print(f"🧩 {len(indices)} total in {low:.2f}–{high:.2f}, selected {len(chosen)}")

# -----------------------------------------------------------------------------
# Plot and export to PDF
# -----------------------------------------------------------------------------
print(f"🖨️ Writing grids to {OUTFILE} ...")

with PdfPages(OUTFILE) as pdf:
    for (low, high, idxs) in selections:
        for idx in idxs:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_grid_image(grids[idx], on_ax=ax)
            r_val = ratings_subset[idx]
            ax.set_title(f"Grid #{idx} | Rating: {r_val:.3f} ∈ [{low:.2f}, {high:.2f})",
                         fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

print(f"✅ Done! Created {OUTFILE} containing {sum(len(s[2]) for s in selections)} grids.")
