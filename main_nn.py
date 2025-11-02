# 2.156 Challenge Problem 1
# Troy Boland, Andrew Castro

# =============================================================================
# ==========================          IMPORTS          ========================
# =============================================================================

from utils_public import *
from train_eval import *
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend, use non-interactive PDF/PNG renderer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm, trange
import seaborn as sns
import pandas as pd



# === CNN branch for spatial input ===
class CNNBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.flatten = nn.Flatten()

        # determine flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 7, 7)
            out_dim = self.forward(dummy).shape[1]
        self.output_dim = out_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.flatten(x)

# === DNN branch for tabular/non-grid features ===
class DNNBranch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# === Fusion head: DNN that takes combined embeddings ===
class CNN_DNN_Hybrid(nn.Module):
    def __init__(self, cnn_channels=1, n_features=10, n_outputs=5):
        super().__init__()
        self.cnn = CNNBranch(cnn_channels)
        self.dnn = DNNBranch(n_features)

        fusion_in_dim = self.cnn.output_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, grid_input, feature_input):
        z_cnn = self.cnn(grid_input)      # spatial features
        z_dnn = self.dnn(feature_input)   # global features
        z = torch.cat([z_cnn, z_dnn], dim=1)
        return self.fusion(z)
    
def train_model(model, X_train, Y_train, X_val, Y_val,
                X2_train=None, X2_val=None,
                num_epochs=10, batch_size=32, learning_rate=0.001, device='cuda'):
    """
    Generic training function that works for:
        - Single-input models (DNN, CNN)
        - Dual-input models (CNN + DNN hybrid)

    Args:
        model: torch.nn.Module
        X_train: main training input (e.g., grids)
        X2_train: secondary input (e.g., tabular features) [optional]
        Y_train: training targets
        X_val, X2_val, Y_val: validation equivalents
    """

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === Dataset & DataLoader ===
    if X2_train is not None:
        # Dual-input dataset
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X2_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X2_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32)
        )
    else:
        # Single-input dataset
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32)
        )



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch in train_bar:
            optimizer.zero_grad()

            # Unpack batch depending on model type
            if X2_train is not None:
                grid_batch = batch[0].to(device)
                feat_batch = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(grid_batch, feat_batch)
            else:
                input = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_bar.set_postfix({'Train Loss': running_loss / len(train_loader.dataset)})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if X2_train is not None:
                    grid_batch, feat_batch, labels = [b.to(device) for b in batch]
                    outputs = model(grid_batch, feat_batch)
                else:
                    inputs, labels = [b.to(device) for b in batch]
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train MSE: {epoch_loss:.4f} - Val MSE: {val_loss:.4f}")

    print("✅ Training complete.")
    return train_losses, val_losses# --- helpers ---
def _ensure_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)  # as_tensor avoids unnecessary copies

def _make_dataset(X1, Y, X2=None):
    X1 = _ensure_tensor(X1)
    Y  = _ensure_tensor(Y)
    if X2 is None:
        return TensorDataset(X1, Y)
    X2 = _ensure_tensor(X2)
    return TensorDataset(X1, X2, Y)

def _match_label_shape(outputs, labels):
    # Align labels to outputs for MSE
    if outputs.dim() == 2 and outputs.size(1) == 1 and labels.dim() == 1:
        labels = labels.unsqueeze(-1)           # [B] -> [B,1]
    elif outputs.dim() == 1 and labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(-1)             # [B,1] -> [B]
    return labels

def train_model(model, X_train, Y_train, X_val, Y_val,
                X2_train=None, X2_val=None,
                num_epochs=10, batch_size=32, learning_rate=1e-3, device='cuda'):

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = _make_dataset(X_train, Y_train, X2_train)
    val_dataset   = _make_dataset(X_val,   Y_val,   X2_val)

    pin = device.startswith('cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=pin)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch in train_bar:
            optimizer.zero_grad()
            if X2_train is not None:
                x1, x2, labels = batch
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(x1, x2)
            else:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)  # <-- fixed (was 'input' vs 'inputs')

            labels = _match_label_shape(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_bar.set_postfix({'Train Loss': running_loss / len(train_loader.dataset)})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if X2_val is not None:
                    x1, x2, labels = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(x1, x2)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)

                labels = _match_label_shape(outputs, labels)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train MSE: {epoch_loss:.4f} - Val MSE: {val_loss:.4f}")

    print("✅ Training complete.")
    return train_losses, val_losses
# =============================================================================
# =========================        TRAIN & EVAL        ======================== 

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

def nn_train_eval(grids_subset, FE_fn, advisor, ratings_subset):
    K = 5
    grids_oh = (np.arange(K) == grids[..., None]).astype(np.float32)   # (N,H,W,K)
    grids_fa = FE_fn(grids, verbose=False).astype(np.float32)
    y = ratings_subset.astype(np.float32)  # (N,) or (N,m)

    X_cnn_tr, X_cnn_te, X_tab_tr, X_tab_te, y_tr, y_te = train_test_split(
        grids_oh, grids_fa, y, test_size=test_size, random_state=seed, shuffle=True
    )

    def to_channels_first(x): return np.transpose(x, (0,3,1,2))  # (N,H,W,K)->(N,K,H,W)

    X_cnn_train_t = torch.as_tensor(to_channels_first(X_cnn_tr), dtype=torch.float32)
    X_cnn_test_t  = torch.as_tensor(to_channels_first(X_cnn_te), dtype=torch.float32)

    X_tab_train_t = torch.as_tensor(X_tab_tr, dtype=torch.float32)
    X_tab_test_t  = torch.as_tensor(X_tab_te, dtype=torch.float32)

    # Ensure (N,1) for single-output; otherwise keep (N,m)
    if y_tr.ndim == 1:
        y_train_t = torch.as_tensor(y_tr, dtype=torch.float32).unsqueeze(-1)  # (N,1)
        y_test_t  = torch.as_tensor(y_te, dtype=torch.float32).unsqueeze(-1)  # (N,1)
        n_out = 1
    else:
        y_train_t = torch.as_tensor(y_tr, dtype=torch.float32)
        y_test_t  = torch.as_tensor(y_te, dtype=torch.float32)
        n_out = y_train_t.shape[1]

    model = CNN_DNN_Hybrid(cnn_channels=K, n_features=X_tab_train_t.shape[1], n_outputs=n_out)

    train_losses, val_losses = train_model(
        model,
        X_train=X_cnn_train_t, Y_train=y_train_t,
        X_val=X_cnn_test_t,    Y_val=y_test_t,
        X2_train=X_tab_train_t, X2_val=X_tab_test_t,
        num_epochs=num_epochs, batch_size=batch_size,
        learning_rate=learning_rate, device=device
    )

    # Predictions (already tensors; no rewrap)
    model = model.to(device)
    preds_train_t = _predict_in_batches(model, X_cnn_train_t, X_tab_train_t, device=device)
    preds_test_t  = _predict_in_batches(model, X_cnn_test_t,  X_tab_test_t,  device=device)

    preds_train = preds_train_t.squeeze(-1).cpu().numpy() if n_out == 1 else preds_train_t.cpu().numpy()
    preds_test  = preds_test_t.squeeze(-1).cpu().numpy()  if n_out == 1 else preds_test_t.cpu().numpy()
    y_train_np  = y_tr if y_tr.ndim == 1 else y_tr.copy()
    y_test_np   = y_te if y_te.ndim == 1 else y_te.copy()

    plot_and_r2(preds_train, preds_test, y_train_np, y_test_np, advisor)
    ...


def fit_plot_predict_nn(grids, ratings, advisor):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) #gets subset of the dataset rated by advisor 0
    print(grids_subset.shape)
    FE_fn = FE_main # Select append_district_counts as our feature engineering function

    NN =  nn_train_eval(grids_subset, FE_fn, advisor, ratings_subset) #feature engineering, split, train, evaluate

    #predictions = merge_predictions(grids, ratings[:,advisor], FE_fn, LR) #merge predictions with actual ratings
    return 


# =============================================================================
# =========================        PREDICTIONS         ========================
# =============================================================================


# Load data
grids = load_grids()
ratings = np.load("datasets/scores.npy")

def predict():
    """
    Runs predictions for all advisors with a progress bar.
    Calls fit_plot_predict() once per advisor and collects predictions.
    """
    all_predictions = []  # store predictions for all advisors

    print("\n🔄 Running advisor predictions...\n")

    # tqdm bar for advisors
    for advisor in tqdm(range(4), desc="Advisors", ncols=80):
        print("advisor count " + str(advisor))
        predictions = fit_plot_predict(grids, ratings, advisor)
        all_predictions.append(predictions)

    print("\n✅ All advisor predictions complete.\n")
    return all_predictions


# Run the pipeline
all_predictions = predict()

# ---- LaTeX PDF Build Section ----
try:
    subprocess.run(["latexmk", "-pdf", "plots.tex"], check=True)
    print("✅ LaTeX PDF rebuilt successfully.")
except FileNotFoundError:
    print("⚠️ LaTeXmk not found. Skipping PDF build.")
except subprocess.CalledProcessError:
    print("❌ LaTeX build failed. Check plots.tex for errors.")