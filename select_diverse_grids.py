#!/usr/bin/env python3
"""
select_diverse_grids.py

Purpose:
    From a batch of approved grids, select the 100 most structurally diverse
    layouts (based on one-hot cityblock distance) and save them to new files.
    Also compares diversity against the full approved set.

Author:
    Troy Boland
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist

# =============================================================================
# HELPERS
# =============================================================================

def onehot_and_flatten(grids, n_classes=5):
    """One-hot encode and flatten (N,7,7) grids into binary feature vectors."""
    grids_oh = (np.arange(n_classes) == grids[..., None]).astype(np.float32)
    return grids_oh.reshape(grids.shape[0], -1)


def diversity_score(grids, N=None):
    """
    Compute normalized diversity score in [0,1] for a batch of categorical grids.

    Parameters
    ----------
    grids : np.ndarray
        (N,7,7) integer grids.
    N : int, optional
        Normalization constant. Defaults to len(grids).
    """
    if N is None:
        N = len(grids)
    grids_flat = onehot_and_flatten(grids)
    n = grids.shape[1]
    D = squareform(pdist(grids_flat, "cityblock"))
    return np.sum(D) / (N**2 * n**2 * 2)


def select_diverse_subset(grids, k=100, verbose=True):
    """
    Select k maximally diverse grids via greedy farthest-point sampling.
    """
    M = len(grids)
    grids_flat = onehot_and_flatten(grids)
    selected = [np.random.randint(M)]
    remaining = set(range(M)) - {selected[0]}

    if verbose:
        print(f"🌱 Starting with random seed grid {selected[0]} out of {M}")
        pbar = tqdm(total=k - 1, desc="Selecting diverse grids")

    for _ in range(k - 1):
        sel_flat = grids_flat[selected]
        rem_flat = grids_flat[list(remaining)]
        dist_to_sel = cdist(rem_flat, sel_flat, metric="cityblock")
        min_dists = dist_to_sel.min(axis=1)

        best_idx = list(remaining)[np.argmax(min_dists)]
        selected.append(best_idx)
        remaining.remove(best_idx)

        if verbose:
            pbar.update(1)
            if len(selected) % 10 == 0:
                pbar.set_postfix({"avg_min_dist": float(np.mean(min_dists))})

    if verbose:
        pbar.close()
        print(f"✅ Selected {len(selected)} diverse grids from {M} candidates.")
    return np.array(selected)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    # ---- Config ----
    INPUT_GRIDS = "selected_grids/approved_grids.npy"
    INPUT_SCORES = "selected_grids/approved_scores.npy"
    OUTPUT_DIR = "selected_grids"
    N_SELECT = 100
    MIN_AVG_SCORE = 0.8  # optional quality filter

    print("🔍 Loading approved grids and scores...")
    grids = np.load(INPUT_GRIDS)
    scores = np.load(INPUT_SCORES)
    print(f"Loaded {len(grids)} grids, each with {scores.shape[1]} advisor scores.")

    # ---- Optional filter by average score ----
    avg_scores = scores.mean(axis=1)
    mask = avg_scores >= MIN_AVG_SCORE
    grids = grids[mask]
    scores = scores[mask]
    print(f"📊 Retained {len(grids)} grids with avg score ≥ {MIN_AVG_SCORE}")

    # ---- Compute baseline diversity for all ----
    print("⚙️ Computing diversity of full approved pool...")
    full_div = diversity_score(grids)
    print(f"🌐 Diversity score (all {len(grids)} grids): {full_div:.6f}")

    # ---- Diversity selection ----
    diverse_idx = select_diverse_subset(grids, k=N_SELECT, verbose=True)
    diverse_grids = grids[diverse_idx]
    diverse_scores = scores[diverse_idx]

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "diverse_100_grids.npy"), diverse_grids)
    np.save(os.path.join(OUTPUT_DIR, "diverse_100_scores.npy"), diverse_scores)
    print("💾 Saved outputs → selected_grids/diverse_100_grids.npy and diverse_100_scores.npy")

    # ---- Compute & Print Diversity Scores ----
    div_score = diversity_score(diverse_grids, N=N_SELECT)
    print(f"\n🌈 Diversity score for selected 100 grids: {div_score:.6f}")
    print(f"📈 Improvement over full pool: {div_score/full_div:.2f}× higher diversity\n")

    # ---- Visualization ----
    print("🖼️ Visualizing selected diverse grids...")
    cols = 10
    rows = N_SELECT // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    fig.suptitle("Top 100 Most Diverse Approved Grids", fontsize=18)

    for i, ax in enumerate(axes.flat):
        g = diverse_grids[i]
        sc = diverse_scores[i]
        ax.imshow(g, cmap="viridis", vmin=0, vmax=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"A0:{sc[0]:.2f}\nA1:{sc[1]:.2f}\nA2:{sc[2]:.2f}\nA3:{sc[3]:.2f}",
            fontsize=8,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "select_diverse_grids":
    main()
