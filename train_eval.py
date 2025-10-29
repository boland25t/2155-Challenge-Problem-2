from utils_public import *
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch

from sklearn.metrics import r2_score

from fe import *
from plot_utils import *
from neural import *

# =============================================================================
# =======================       TRAINING/EVAL       ===========================
# =============================================================================

def LR_split_train_eval(grids, FE_fn, advisor, ratings_subset):
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


import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---- helper: fast, memory-safe prediction in batches ----
def _predict_in_batches(model, X1, X2=None, batch_size=1024, device="cpu"):
    model.eval()
    preds = []
    if X2 is None:
        ds = TensorDataset(X1)
    else:
        ds = TensorDataset(X1, X2)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            if X2 is None:
                x1 = batch[0].to(device)
                yhat = model(x1)
            else:
                x1 = batch[0].to(device)
                x2 = batch[1].to(device)
                yhat = model(x1, x2)
            preds.append(yhat.detach().cpu())
    return torch.cat(preds, dim=0)

def nn_train_eval(
    grids, FE_fn, advisor, ratings_subset,
    test_size=0.25, seed=42, device='cpu',
    num_epochs=20, batch_size=64, learning_rate=1e-3,
    save_model_path=None
):
    """
    Trains CNN+DNN hybrid on one-hot 7x7 grids + engineered features,
    returns losses/metrics, and saves R^2 plots identical to your LR plots.
    """

    # --- 1) Build inputs ---
    K = 5  # number of classes (adjust if needed)
    # (N, H, W, K)
    grids_oh = (np.arange(K) == grids[..., None]).astype(np.float32)

    # Engineered features (N, d)
    grids_fa = FE_fn(grids, verbose=False).astype(np.float32)

    # Targets (N, m) or (N,)
    y = ratings_subset.astype(np.float32)

    # --- 2) Train/Test split for all aligned arrays ---
    X_cnn_train, X_cnn_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
        grids_oh, grids_fa, y,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # --- 3) Prepare tensors (channels-first for CNN) ---
    def to_cnn_channels_first(x):  # (N, H, W, K) -> (N, K, H, W)
        return np.transpose(x, (0, 3, 1, 2))

    X_cnn_train_t = torch.tensor(to_cnn_channels_first(X_cnn_train), dtype=torch.float32)
    X_cnn_test_t  = torch.tensor(to_cnn_channels_first(X_cnn_test),  dtype=torch.float32)

    X_tab_train_t = torch.tensor(X_tab_train, dtype=torch.float32)
    X_tab_test_t  = torch.tensor(X_tab_test,  dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # --- 4) Build model & train ---
    model = CNN_DNN_Hybrid(
        cnn_channels=K,
        n_features=X_tab_train.shape[1],
        n_outputs=1 if y_train_t.ndim == 1 else y_train_t.shape[1]
    )

    train_losses, val_losses = train_model(
        model,
        X_train=X_cnn_train_t, Y_train=y_train_t,
        X_val=X_cnn_test_t, Y_val=y_test_t,
        X2_train=X_tab_train_t, X2_val=X_tab_test_t,
        num_epochs=num_epochs, batch_size=batch_size,
        learning_rate=learning_rate, device=device
    )

    # --- 5) Predict on train/test in batches (handles CPU/GPU) ---
    model = model.to(device)
    preds_train_t = _predict_in_batches(model, X_cnn_train_t, X_tab_train_t, batch_size=1024, device=device)
    preds_test_t  = _predict_in_batches(model, X_cnn_test_t,  X_tab_test_t,  batch_size=1024, device=device)

    # Ensure 1D for single-output to match your plotter expectations
    preds_train = preds_train_t.squeeze(-1).cpu().numpy()
    preds_test  = preds_test_t.squeeze(-1).cpu().numpy()
    y_train_np  = y_train_t.squeeze(-1).cpu().numpy()
    y_test_np   = y_test_t.squeeze(-1).cpu().numpy()

    # --- 6) Plot & print R^2 using your existing utility (identical look) ---
    # Reuses your plot_and_r2(...) exactly as-is
    plot_and_r2(preds_train, preds_test, y_train_np, y_test_np, advisor)

    # --- 7) Compute numeric R^2 as return values, too ---
    r2_train = r2_score(y_train_np, preds_train)
    r2_test  = r2_score(y_test_np,  preds_test)

    # --- 8) (Optional) Save the trained model ---
    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f"💾 Saved model to: {save_model_path}")

    # --- 9) Return everything useful for downstream use ---
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "preds_train": preds_train,
        "preds_test": preds_test,
        "y_train": y_train_np,
        "y_test": y_test_np,
        "splits": {
            "X_cnn_train": X_cnn_train_t,
            "X_cnn_test":  X_cnn_test_t,
            "X_tab_train": X_tab_train_t,
            "X_tab_test":  X_tab_test_t,
        }
    }


    

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

    LR = LR_split_train_eval(grids_subset, FE_fn, advisor, ratings_subset) #feature engineering, split, train, evaluate

    #predictions = merge_predictions(grids, ratings[:,advisor], FE_fn, LR) #merge predictions with actual ratings
    return 

def fit_plot_predict_nn(grids, ratings, advisor):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) #gets subset of the dataset rated by advisor 0
    print(grids_subset.shape)
    FE_fn = FE_main # Select append_district_counts as our feature engineering function

    NN =  nn_train_eval(grids_subset, FE_fn, advisor, ratings_subset) #feature engineering, split, train, evaluate

    #predictions = merge_predictions(grids, ratings[:,advisor], FE_fn, LR) #merge predictions with actual ratings
    return 