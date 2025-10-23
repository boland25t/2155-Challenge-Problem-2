from utils_public import *
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from fe import *
from plot_utils import *


# =============================================================================
# =======================       TRAINING/EVAL       ===========================
# =============================================================================

def FE_split_train_eval(grids, FE_fn, advisor, ratings_subset):
    """
    Perform feature engineering, split data, train a LinearRegression model,
    clip predictions to [0, 1], and evaluate with R² + plots.

    Parameters
    ----------
    grids : np.ndarray
        Full grid dataset (N, 7, 7).
    FE_fn : callable
        Feature engineering function returning feature array.
    advisor : int
        Advisor index (used for labeling plots).
    ratings_subset : np.ndarray
        Ratings vector (1-D) for this advisor, matching grids.

    Returns
    -------
    LR : sklearn.linear_model.LinearRegression
        Fitted regression model.
    """

    # === Feature Engineering ===
    grids_fa = FE_fn(grids, verbose=False)

    # === Train/Test Split ===
    grids_train, grids_test, ratings_train, ratings_test = train_test_split(
        grids_fa, ratings_subset, test_size=0.25, random_state=42
    )

    # === Train Linear Model ===
    LR = LinearRegression()
    LR.fit(grids_train, ratings_train)

    # === Predictions ===
    preds_train = LR.predict(grids_train)
    preds_test  = LR.predict(grids_test)

    # === BLUF: Clip predictions to [0, 1] to keep valid rating range ===
    preds_train = np.clip(preds_train, 0, 1)
    preds_test  = np.clip(preds_test, 0, 1)

    # === Plot + Evaluate ===
    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor)

    return LR



def merge_predictions(grids, ratings, FE_fn, predictor): #Combine predictions and real ratings
    full_dataset_FA = FE_fn(grids) #feature engineering on full dataset
    predictions = predictor.predict(full_dataset_FA) #predict on full dataset

    mask = np.where(~np.isnan(ratings)) #get the indices of the rated grids
    predictions[mask] = ratings[mask] #replace the predictions with the actual ratings where available
    return predictions

def fit_plot_predict(grids, ratings, advisor):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) #gets subset of the dataset rated by advisor 0
    print(grids_subset.shape)
    FE_fn = FE_main # Select append_district_counts as our feature engineering function

    LR = FE_split_train_eval(grids_subset, FE_fn, advisor, ratings_subset) #feature engineering, split, train, evaluate

    #predictions = merge_predictions(grids, ratings[:,advisor], FE_fn, LR) #merge predictions with actual ratings
    return 