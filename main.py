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

# =============================================================================
# =========================        PREDICTIONS         ========================
# =============================================================================
grids = load_grids()

ratings = np.load("datasets/scores.npy")

def predict():
    all_predictions = [] #empty list to hold predictions
    for i in range(0,4): #loop over four advisors
        predictions = fit_plot_predict(grids, ratings, i) #call the above function
        all_predictions.append(predictions) #append predictions

predict()






