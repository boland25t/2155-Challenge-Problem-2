#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_diverse_candidates.py
Use existing advisor .pt models (fully saved) to score new grids,
merge with approved ones, and select the most diverse subset.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from fe import FE_main
from select_diverse_grids import onehot_and_flatten, diversity_score, select_diverse_subset

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = "cpu"
NUM_CLASSES = 5
MODEL_DIR = "saved_models"
OUTDIR = "selected_grids"
N_GEN = 5000
MIN_SCORE = 0.8
N_SELECT = 100

# =============================================================================
# GRID GENERATION
# =============================================================================
def generate_structured_grids(N=5000, n_classes=NUM_CLASSES):
    """Generate structurally varied grids."""
    grids = np.zeros((N, 7, 7), dtype=int)
    for i in range(N):
        mode = np.random.choice(["uniform", "blocky", "striped", "checker", "gradient"])
        g = np.random.randint(0, n_classes, (7, 7))
        if mode == "blocky":
            block_val = np.random.randint(0, n_classes)
            x0, y0 = np.random.randint(0, 5), np.random.randint(0, 5)
            g[x0:x0+3, y0:y0+3] = block_val
        elif mode == "striped":
            g[::2, :] = np.random.randint(0, n_classes)
        elif mode == "checker":
            g = (np.indices((7, 7)).sum(axis=0) + np.random.randint(0, n_classes)) % n_classes
        elif mode == "gradient":
            g = np.tile(np.arange(7) % n_classes, (7, 1))
        grids[i] = g
    return grids

# =============================================================================
# MODEL LOADING AND PREDICTION
# =============================================================================
def load_advisor_models(model_dir=MODEL_DIR):
    """Load all advisor models (.pt files containing entire models)."""
    models = []
    for i in range(4):
        path = os.path.join(model_dir, f"advisor_{i}_model.pt")
        assert os.path.exists(path), f"❌ Missing model: {path}"
        model = torch.load(path, map_location=DEVICE)
        model.eval()
        models.append(model)
        print(f"✅ Loaded advisor {i} from {path}")
    return models

def predict_with_models(models, grids, features):
    """Run each advisor model on all grids and return an (N,4) matrix of predictions."""
    grids_onehot = np.transpose((np.arange(NUM_CLASSES) == grids[..., None]).astype(np.float32),
                                (0, 3, 1, 2))
    X_cnn = torch.tensor(grids_onehot, dtype=torch.float32, device=DEVICE)
    X_tab = torch.tensor(features, dtype=torch.float32, device=DEVICE)

    preds = []
    for i, model in enumerate(models):
        with torch.no_grad():
            y = model(X_cnn, X_tab).squeeze().cpu().numpy()
        preds.append(y)
    return np.stack(preds, axis=1)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🎲 Generating candidate grids...")
    new_grids = generate_structured_grids(N_GEN)

    print("⚙️  Running feature extraction (FE_main)...")
    new_features = FE_main(new_grids)

    print("🤖 Loading advisor models...")
    models = load_advisor_models()

    print("🔮 Predicting scores for new grids...")
    new_scores = predict_with_models(models, new_grids, new_features)

    avg_scores = new_scores.mean(axis=1)
    keep = avg_scores >= MIN_SCORE
    kept_grids = new_grids[keep]
    kept_scores = new_scores[keep]
    print(f"✅ {keep.sum()} of {len(new_grids)} meet avg ≥ {MIN_SCORE}")

    # Merge with existing approved grids
    old_grids = np.load(f"{OUTDIR}/approved_grids.npy")
    old_scores = np.load(f"{OUTDIR}/approved_scores.npy")
    merged_grids = np.concatenate([old_grids, kept_grids], axis=0)
    merged_scores = np.concatenate([old_scores, kept_scores], axis=0)
    print(f"📦 Total merged pool: {len(merged_grids)} grids")

    # Diversity improvement
    print("📊 Calculating diversity...")
    base_div = diversity_score(merged_grids)
    print(f"🌐 Diversity (merged pool): {base_div:.6f}")

    diverse_idx = select_diverse_subset(merged_grids, k=N_SELECT)
    diverse_grids = merged_grids[diverse_idx]
    diverse_scores = merged_scores[diverse_idx]
    div_score = diversity_score(diverse_grids)
    print(f"🌈 Diversity (selected subset): {div_score:.6f}")
    print(f"📈 Improvement factor: {div_score / base_div:.2f}×")

    os.makedirs(OUTDIR, exist_ok=True)
    np.save(f"{OUTDIR}/diverse_augmented_grids.npy", diverse_grids)
    np.save(f"{OUTDIR}/diverse_augmented_scores.npy", diverse_scores)
    print("💾 Saved diverse augmented grids in:", OUTDIR)

if __name__ == "__main__":
    main()
