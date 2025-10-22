from utils_public import *
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from fe import *
from plot_utils import *

from sklearn.model_selection import train_test_splits
import pandas as pd

# =============================================================================
# =======================       TRAINING/EVAL       ===========================
# =============================================================================

def FE_split_train_eval(grids, FE_fn, advisor, ratings_subset): #feature engineering, split, train, evaluate
    grids_fa = FE_fn(grids, verbose=False) #call the provided feature engineering function on the grids
    
    grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_fa, ratings_subset) #split

    LR = LinearRegression() #initialize
    LR.fit(grids_train, ratings_train) #fit

    preds_train = LR.predict(grids_train) #predict on the train set
    preds_test = LR.predict(grids_test) #predict on the test set
    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor) #plot and calculate R2
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