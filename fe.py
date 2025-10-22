
from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd

from scipy.ndimage import label

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score


# =============================================================================
# =======================       FEATURE ENGINEERING       =====================
# =============================================================================

# FE_main is passed to FE_split_train_eval
# Returns flattened grids with features appended



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

def FE_main(grids):
    
    grids_flat = grids.reshape(-1, 49)

    # ---- Step 2: Compute both feature sets via helper functions ----
    district_features = append_district_counts(grids)  # (n_grids, 5)
    contig_features   = append_contig_counts(grids)    # (n_grids, 5)
    
    # ---- Step 3: Combine all features ----
    features_full = np.hstack([grids_flat, district_features, contig_features])

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
