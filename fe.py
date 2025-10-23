from utils_public import *
from train_eval import *

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import entropy

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for headless execution

import time

# =============================================================================
# =======================       FEATURE ENGINEERING       =====================
# =============================================================================
# FE_main is passed to FE_split_train_eval
# Returns flattened grids with features appended

# =============================================================================
# =======================       BASE FEATURE FUNCTIONS     ====================
# =============================================================================

def compute_zone_adjacencies(grid):
    """
    Compute upper-triangle adjacency counts between zone types.

    Returns (10,) array for pairs:
    (0,1),(0,2),(0,3),(0,4),
    (1,2),(1,3),(1,4),
    (2,3),(2,4),
    (3,4)
    """
    grid = grid.astype(int)
    adj_counts = np.zeros((5, 5), dtype=int)

    # Horizontal adjacencies
    left_cells, right_cells = grid[:, :-1], grid[:, 1:]
    for a, b in zip(left_cells.flatten(), right_cells.flatten()):
        if a != b:
            adj_counts[a, b] += 1
            adj_counts[b, a] += 1

    # Vertical adjacencies
    top_cells, bottom_cells = grid[:-1, :], grid[1:, :]
    for a, b in zip(top_cells.flatten(), bottom_cells.flatten()):
        if a != b:
            adj_counts[a, b] += 1
            adj_counts[b, a] += 1

    pairs = [(0,1),(0,2),(0,3),(0,4),
             (1,2),(1,3),(1,4),
             (2,3),(2,4),
             (3,4)]
    return np.array([adj_counts[i, j] for i, j in pairs], dtype=int)


def compute_zone_perimeters(grid):
    """
    Compute total perimeter (exposed edges) for each zone type in a 7x7 grid.
    Returns (5,) array of perimeters per zone label.
    """
    grid = grid.astype(int)
    perims = np.zeros(5, dtype=int)
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=-1)

    for k in range(5):
        mask = (padded == k)
        up    = mask[1:-1, 1:-1] & (padded[:-2, 1:-1] != k)
        down  = mask[1:-1, 1:-1] & (padded[2:,  1:-1] != k)
        left  = mask[1:-1, 1:-1] & (padded[1:-1, :-2] != k)
        right = mask[1:-1, 1:-1] & (padded[1:-1, 2:]  != k)
        perims[k] = np.sum(up) + np.sum(down) + np.sum(left) + np.sum(right)

    return perims


def append_contig_counts(grids):
    """
    Count contiguous (connected) regions per zone label for each grid.
    Returns array (n_grids, 5)
    """
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])
    contig_features = []
    for grid in grids:
        per_label_counts = []
        for k in range(5):
            mask = (grid == k)
            _, num_features = label(mask, structure=structure)
            per_label_counts.append(num_features)
        contig_features.append(per_label_counts)
    return np.array(contig_features)


def append_district_counts(grids):
    """
    Count total occurrences of each district label per grid.
    Returns array (n_grids, 5)
    """
    grids_flat = grids.reshape(-1, 49)
    counts = [np.sum(grids_flat == k, axis=1) for k in range(5)]
    return np.stack(counts).T


# =============================================================================
# ======================   SHANNON ENTROPY FEATURE   ==========================
# =============================================================================

def FE_entropy(grids, verbose=False):
    """
    Compute normalized Shannon entropy (0–1) for each grid.

    Parameters
    ----------
    grids : np.ndarray of shape (N,H,W) with integer values 0–3
    verbose : bool
    """
    n_classes = 4
    grids = np.asarray(grids).astype(int)  # 🔧 <-- ensure integer dtype
    entropies = np.zeros((grids.shape[0], 1))

    for i, g in enumerate(grids):
        counts = np.bincount(g.ravel(), minlength=n_classes)
        p = counts / np.sum(counts)
        H = entropy(p, base=np.e)
        H_norm = H / np.log(n_classes)
        entropies[i, 0] = H_norm

    if verbose:
        print(f"[FE_entropy] Mean entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
    return entropies

# =============================================================================
# ======================     FRACTAL DIMENSION FEATURE     ====================
# =============================================================================
from scipy.ndimage import binary_erosion

def FE_fractal_dimension(grids, verbose=False):
    """
    Estimate the fractal dimension (roughness) for each zone in each grid
    using a box-counting approach.

    Returns
    -------
    np.ndarray : shape (N, 5)
        Fractal dimension per zone label (0–4)
    """
    from tqdm import tqdm
    grids = np.asarray(grids).astype(int)
    n_grids = grids.shape[0]
    sizes = np.array([1, 2, 3])  # box sizes
    fractal_feats = np.zeros((n_grids, 5))

    for i, grid in enumerate(tqdm(grids, desc="Computing fractal dimension")):
        for label_val in range(5):
            mask = (grid == label_val)
            counts = []
            for s in sizes:
                new_shape = (mask.shape[0] // s, s, mask.shape[1] // s, s)
                reduced = mask[:new_shape[0]*s, :new_shape[2]*s].reshape(new_shape)
                boxes = reduced.max(axis=(1,3))
                counts.append(np.sum(boxes))
            if len(set(counts)) > 1 and np.all(np.array(counts) > 0):
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                fractal_feats[i, label_val] = -coeffs[0]
            else:
                fractal_feats[i, label_val] = 0.0

    if verbose:
        print(f"[FE_fractal_dimension] Mean: {np.mean(fractal_feats):.3f} ± {np.std(fractal_feats):.3f}")
    return fractal_feats



# =============================================================================
# ======================        MORAN'S I FEATURE           ====================
# =============================================================================
def FE_morans_I(grids, verbose=False):
    """
    Compute global Moran's I for each grid (spatial autocorrelation).

    Returns
    -------
    np.ndarray : shape (N, 1)
    """
    from tqdm import tqdm
    grids = np.asarray(grids).astype(float)
    N = grids.shape[1] * grids.shape[2]
    morans = np.zeros((grids.shape[0], 1))

    # Precompute rook adjacency weights (7x7 fixed grid)
    def rook_weights(H, W):
        w = np.zeros((H*W, H*W))
        for i in range(H):
            for j in range(W):
                idx = i*W + j
                for (di, dj) in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W:
                        w[idx, ni*W + nj] = 1
        return w

    W = rook_weights(7, 7)
    W_sum = np.sum(W)

    for k, grid in enumerate(tqdm(grids, desc="Computing Moran's I")):
        x = grid.flatten()
        x_bar = np.mean(x)
        num = np.sum(W * np.outer(x - x_bar, x - x_bar))
        den = np.sum((x - x_bar)**2)
        morans[k, 0] = (N / W_sum) * (num / den) if den > 0 else 0.0

    if verbose:
        print(f"[FE_morans_I] Mean Moran's I: {np.mean(morans):.3f}")
    return morans

# =============================================================================
# ===================     AVERAGE & LARGEST ZONE AREA       ====================
# =============================================================================
from scipy.ndimage import label

def FE_zone_areas(grids, verbose=False):
    """
    Compute average and largest contiguous zone area per grid.

    Returns
    -------
    np.ndarray : shape (N, 2)
        [:,0] = average contiguous zone area
        [:,1] = normalized largest zone area (0–1)
    """
    from tqdm import tqdm
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])
    n = grids.shape[0]
    feats = np.zeros((n, 2))
    total_area = grids.shape[1] * grids.shape[2]

    for i, grid in enumerate(tqdm(grids.astype(int), desc="Computing zone areas")):
        areas = []
        for k in range(5):
            mask = (grid == k)
            labeled, num = label(mask, structure=structure)
            if num > 0:
                counts = np.bincount(labeled.ravel())[1:]  # skip background
                areas.extend(counts)
        if len(areas) > 0:
            feats[i, 0] = np.mean(areas)
            feats[i, 1] = np.max(areas) / total_area
        else:
            feats[i, :] = 0.0

    if verbose:
        print(f"[FE_zone_areas] Mean avg area: {np.mean(feats[:,0]):.2f}, mean largest: {np.mean(feats[:,1]):.2f}")
    return feats

# =============================================================================
# ======================     EDGE-CONTACT ZONE FEATURE     ====================
# =============================================================================
def FE_edge_contact(grids, mode="binary", verbose=False):
    """
    Quantify which zone types contact the grid border.

    Parameters
    ----------
    grids : np.ndarray
        (N, H, W) array of integer zone labels.
    mode : str
        "binary"  → 1 if a zone type touches any edge, else 0.
        "count"   → number of edge cells of that type (normalized by perimeter).
    verbose : bool

    Returns
    -------
    np.ndarray : shape (N, 5)
        Feature per zone label (0–4)
    """
    from tqdm import tqdm
    grids = np.asarray(grids).astype(int)
    n = grids.shape[0]
    features = np.zeros((n, 5))
    H, W = grids.shape[1], grids.shape[2]
    edge_perimeter = 2 * (H + W) - 4

    for i, grid in enumerate(tqdm(grids, desc="Computing edge contacts")):
        # Extract edge cells
        edge_cells = np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ])
        if mode == "binary":
            # Mark 1 if that label appears on edge
            for k in range(5):
                features[i, k] = int(k in edge_cells)
        elif mode == "count":
            # Count normalized frequency of each label on edge
            for k in range(5):
                features[i, k] = np.sum(edge_cells == k) / edge_perimeter

    if verbose:
        if mode == "binary":
            print(f"[FE_edge_contact] Mean zones touching edge: {np.mean(np.sum(features, axis=1)):.2f}")
        else:
            print(f"[FE_edge_contact] Mean edge contact fraction per label: {np.mean(features, axis=0)}")
    return features
# =============================================================================
# =======================       MAIN FE PIPELINE         =====================  

def FE_main(grids, verbose=False):
    """
    Main feature extraction pipeline.
    Returns flattened grids with appended engineered features.
    """
    start = time.time()
    grids = np.asarray(grids)
    if grids.ndim == 2 and grids.shape == (7, 7):
        grids = grids[None, :, :]

    print(f"  → FE_main: processing {len(grids)} grids")

    # --- Base features ---
    district_features = append_district_counts(grids)
    contig_features   = append_contig_counts(grids)
    print(f"  ✓ Base features computed in {time.time() - start:.2f}s")

    # --- Perimeter features ---
    perim_start = time.time()
    perim_features = [compute_zone_perimeters(g) for g in tqdm(grids, desc="Computing perimeters")]
    print(f"  ✓ Perimeters done in {time.time() - perim_start:.2f}s")

    # --- Adjacency features ---
    adj_start = time.time()
    adj_features = [compute_zone_adjacencies(g) for g in tqdm(grids, desc="Computing adjacencies")]
    print(f"  ✓ Adjacencies done in {time.time() - adj_start:.2f}s")

    # --- Shannon Entropy ---
    entropy_start = time.time()
    entropy_features = [FE_entropy(np.array([g]))[0] for g in tqdm(grids, desc="Computing entropy")]
    entropy_features = np.array(entropy_features)
    print(f"  ✓ Entropy done in {time.time() - entropy_start:.2f}s")

    # --- Fractal Dimension ---
    fractal_start = time.time()
    fractal_features = FE_fractal_dimension(grids)
    print(f"  ✓ Fractal dimension done in {time.time() - fractal_start:.2f}s")

    # --- Moran's I ---
    moran_start = time.time()
    moran_features = FE_morans_I(grids)
    print(f"  ✓ Moran's I done in {time.time() - moran_start:.2f}s")

    # --- Zone Area Metrics ---
    zonearea_start = time.time()
    zone_area_features = FE_zone_areas(grids)
    print(f"  ✓ Zone area features done in {time.time() - zonearea_start:.2f}s")

    # --- Edge Contact Features ---
    edge_start = time.time()
    edge_features = FE_edge_contact(grids, mode="binary")  # or "count"
    print(f"  ✓ Edge contact features done in {time.time() - edge_start:.2f}s")


    # --- Combine all features ---
    grids_flat = grids.reshape(-1, 49)
    features_full = np.hstack([
        grids_flat,
        district_features,
        contig_features,
        perim_features,
        adj_features,
        entropy_features,
        fractal_features,
        moran_features,
        zone_area_features,
        edge_features
    ])


    print(f"  ✓ FE_main total time {time.time() - start:.2f}s\n")
    return features_full


# =============================================================================
# =======================       TESTING SPACE             =====================
# =============================================================================
if __name__ == "__main__":
    # Sanity check
    grids_test = np.array([
        np.zeros((7,7)),                     # uniform grid
        np.tile(np.arange(7) % 4, (7,1)),    # patterned
        np.random.randint(0,4,(7,7))         # random
    ])
    feats = FE_main(grids_test, verbose=True)
    print("Entropy values (last column):", feats[:, -1])
