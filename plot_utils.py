from utils_public import *
import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import subprocess

# =============================================================================
# =========================     PLOTTING UTILITIES     ========================
# =============================================================================


def plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor): 
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)
    #Calculates 
    plt.figure(figsize=(6,3))
    plt.scatter(ratings_train, preds_train, label='Train Set Preds', s=3, c = "#BBBBBB") #train set in gray
    plt.scatter(ratings_test, preds_test, label='Test Set Preds', s=5, c = "#DC267F") #test set in magenta
    plt.plot([0,1], [0,1], label="Target", linewidth=3, c="k") #target line in black

    #Set axis labels and title
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Advisor {advisor} Predictions")

    #Turn off top and right spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend() #Display legend
    
    r2_train = r2_score(ratings_train, preds_train)
    r2_test = r2_score(ratings_test, preds_test)

        # Add R² text box
    textstr = (
        f"$R^2_{{train}}$ = {r2_train:.3f}\n"
        f"$R^2_{{test}}$ = {r2_test:.3f}"
    )
    plt.text(
        0.97, 0.03, textstr,
        transform=ax.transAxes,
        fontsize=10,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    fname = f"{outdir}/plot_{advisor}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {fname}")

    #Calculate R2 score for train and test sets
    print(f"Advisor {advisor} Train Set R2 score: {r2_score(ratings_train, preds_train)}") 
    print(f"Advisor {advisor} Test Set R2 score: {r2_score(ratings_test, preds_test)}")
    

    