
from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd

from scipy.ndimage import label

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend, use non-interactive PDF/PNG renderer

# =============================================================================
# =======================       FEATURE ENGINEERING       =====================
# =============================================================================

# FE_main is passed to FE_split_train_eval
# Returns flattened grids with features appended

def compute_zone_adjacencies(grid):
    """
    Compute upper-triangle adjacency counts between zone types.

    Parameters
    ----------
    grid : np.ndarray
        7x7 array with integer zone labels {0,1,2,3,4}.

    Returns
    -------
    np.ndarray
        (10,) array corresponding to pairs:
        (0,1), (0,2), (0,3), (0,4),
        (1,2), (1,3), (1,4),
        (2,3), (2,4),
        (3,4)
    """
    grid = grid.astype(int)

    adj_counts = np.zeros((5, 5), dtype=int)

    # Horizontal adjacencies
    left_cells = grid[:, :-1]
    right_cells = grid[:, 1:]
    for a, b in zip(left_cells.flatten(), right_cells.flatten()):
        if a != b:
            adj_counts[a, b] += 1
            adj_counts[b, a] += 1

    # Vertical adjacencies
    top_cells = grid[:-1, :]
    bottom_cells = grid[1:, :]
    for a, b in zip(top_cells.flatten(), bottom_cells.flatten()):
        if a != b:
            adj_counts[a, b] += 1
            adj_counts[b, a] += 1

    # Extract the upper-triangle unique pairs
    pairs = [(0,1),(0,2),(0,3),(0,4),
             (1,2),(1,3),(1,4),
             (2,3),(2,4),
             (3,4)]
    upper = np.array([adj_counts[i, j] for i, j in pairs], dtype=int)
    return upper


def compute_zone_perimeters(grid):
    """
    Compute total perimeter (exposed edges) for each zone type in a 7x7 grid.

    Parameters
    ----------
    grid : np.ndarray
        7x7 array with integer zone labels {0,1,2,3,4}.

    Returns
    -------
    np.ndarray
        (5,) array: total perimeter length per zone label.
    """
    grid = grid.astype(int)

    h, w = grid.shape
    perims = np.zeros(5, dtype=int)

    # Pad the grid with a border of -1 to handle edges
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=-1)

    for k in range(5):
        mask = (padded == k)

        # Compare with four neighbors: up, down, left, right
        up    = mask[1:-1, 1:-1] & (padded[:-2, 1:-1] != k)
        down  = mask[1:-1, 1:-1] & (padded[2:,  1:-1] != k)
        left  = mask[1:-1, 1:-1] & (padded[1:-1, :-2] != k)
        right = mask[1:-1, 1:-1] & (padded[1:-1, 2:]  != k)

        # Each True represents one exposed edge
        perims[k] = np.sum(up) + np.sum(down) + np.sum(left) + np.sum(right)

    return perims



def append_contig_counts(grids): #performs the feature engineering to add contiguous counts
    grids_flat = grids.reshape(-1, 49) #first flatten the grids
    
    # Build 4-connectivity structure
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])

    # Compute contiguous counts for each zoning label per grid
    contig_features = []
    for grid in grids:
        per_label_counts = []
        for k in range(5):
            mask = (grid == k)
            _, num_features = label(mask, structure=structure)
            per_label_counts.append(num_features)
        contig_features.append(per_label_counts)

    contig_features = np.array(contig_features)  # (n_grids, 5)
    # print(contig_features)
    return contig_features

def append_district_counts(grids): #performs the feature engineering to add district counts
    grids_flat = grids.reshape(-1, 49) #first flatten the grids

    counts = [np.sum(grids_flat==0, axis=1), 
              np.sum(grids_flat==1, axis=1), 
              np.sum(grids_flat==2, axis=1), 
              np.sum(grids_flat==3, axis=1), 
              np.sum(grids_flat==4, axis=1)] #list of 5 length n_grids arrays containing counts of each district
    features = np.stack(counts).T #stack and transpose counts to get n_grids x 5 array
    return features #stack the features horizontally with the flattened grids

import time

def FE_main(grids, verbose=False):
    start = time.time()
    grids = np.asarray(grids)
    if grids.ndim == 2 and grids.shape == (7,7):
        grids = grids[None, :, :]
    grids_flat = grids.reshape(-1, 49)

    print(f"  → FE_main: processing {len(grids)} grids")

    district_features = append_district_counts(grids)
    contig_features   = append_contig_counts(grids)
    print(f"  ✓ Base features computed in {time.time() - start:.2f}s")

    perim_start = time.time()
    perim_features = [compute_zone_perimeters(g) for g in grids]
    print(f"  ✓ Perimeters done in {time.time() - perim_start:.2f}s")

    adj_start = time.time()
    adj_features = [compute_zone_adjacencies(g) for g in grids]
    print(f"  ✓ Adjacencies done in {time.time() - adj_start:.2f}s")

    features_full = np.hstack([
        grids_flat,
        district_features,
        contig_features,
        perim_features,
        adj_features
    ])
    print(f"  ✓ FE_main total time {time.time() - start:.2f}s\n")
    return features_full

# =============================================================================
# =======================       TESTING SPACE             =====================
# =============================================================================

"""
i=4
grid = load_grids()[i]
_ = print(grid)

_ = print(append_contig_counts(np.array([grid])))
"""
