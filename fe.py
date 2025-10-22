
from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score


# =============================================================================
# =======================       FEATURE ENGINEERING       =====================
# =============================================================================

# FE_main is passed to FE_split_train_eval
# Returns flattened grids with features appended

def FE_main():
    return

"""
BLUF: Converts each 7×7 district grid into a 54-length feature row by
flattening the 49 cells and appending 5 engineered features: the counts
of labels 0–4. Output shape = (n_grids, 54).

Input
-----
grids : array-like
    Either (n_grids, 7, 7) or (n_grids, 49). Values should be in {0,1,2,3,4}.

Process (vectorized; O(n_grids * 49))
-------------------------------------
1) Flatten to (n_grids, 49) via `grids.reshape(-1, 49)`.
2) For each label k in {0..4}, compute counts per grid:
   `np.sum(grids_flat == k, axis=1)` → shape (n_grids,).
3) Stack the 5 count vectors and transpose → (n_grids, 5).
4) Horizontally concatenate with the flattened cells → (n_grids, 54).

Return
------
np.ndarray of shape (n_grids, 54):
    [49 flattened cells | count_0 | count_1 | count_2 | count_3 | count_4]

Examples
--------
• Two grids:
  - Grid A: 47 zeros, one '1', one '4' → counts [47, 1, 0, 0, 1]
  - Grid B: all '2' (49 of them)        → counts [0, 0, 49, 0, 0]
  Output rows end with those count vectors respectively.

• One grid with totals c0=10, c1=9, c2=11, c3=8, c4=11 (sum=49):
  The output row ends with [10, 9, 11, 8, 11].

Notes
-----
• Labels outside 0..4 are not included in these five counts.
• Dtype follows the input; counts are integers.
"""
def append_district_counts(grids): #performs the feature engineering to add district counts
    grids_flat = grids.reshape(-1, 49) #first flatten the grids

    counts = [np.sum(grids_flat==0, axis=1), 
              np.sum(grids_flat==1, axis=1), 
              np.sum(grids_flat==2, axis=1), 
              np.sum(grids_flat==3, axis=1), 
              np.sum(grids_flat==4, axis=1)] #list of 5 length n_grids arrays containing counts of each district
    features = np.stack(counts).T #stack and transpose counts to get n_grids x 5 array
    return np.hstack([grids_flat, features]) #stack the features horizontally with the flattened grids