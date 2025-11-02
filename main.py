# 2.156 Challenge Problem 2
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
    return train_losses, val_losses

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