import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from main_nn import CNN_DNN_Hybrid, FE_main
from utils_public import load_grids

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
print("FLAG")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ADVISORS = 4
NUM_CLASSES = 5
BATCH_SIZE = 256
THRESHOLD = 0.8

MODEL_DIR = "saved_models"
OUTPUT_DIR = "selected_grids"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. LOAD UNGRADED GRIDS
# --------------------------------------------------------------------------
new_grids = load_grids()  # or load_grids("datasets/grids_ungraded.npy")
print(f"Loaded {new_grids.shape[0]} ungraded grids")

# --------------------------------------------------------------------------
# 2. PREPARE CNN + FEATURE INPUTS
# --------------------------------------------------------------------------
grids_oh = (np.arange(NUM_CLASSES) == new_grids[..., None]).astype(np.float32)
grids_cnn = torch.as_tensor(np.transpose(grids_oh, (0, 3, 1, 2)), dtype=torch.float32)
grids_fa = FE_main(new_grids, verbose=False).astype(np.float32)
grids_tab = torch.as_tensor(grids_fa, dtype=torch.float32)

# --------------------------------------------------------------------------
# 3. LOAD MODELS
# --------------------------------------------------------------------------
models = []
for i in range(NUM_ADVISORS):
    model_path = os.path.join(MODEL_DIR, f"advisor_{i}_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    model = CNN_DNN_Hybrid(
        cnn_channels=NUM_CLASSES,
        n_features=grids_tab.shape[1],
        n_outputs=1
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    models.append(model)
    print(f"✅ Loaded advisor {i} model from {model_path}")

# --------------------------------------------------------------------------
# 4. RUN PREDICTIONS
# --------------------------------------------------------------------------
dataset = TensorDataset(grids_cnn, grids_tab)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

advisor_preds = [torch.empty((0, 1)) for _ in range(NUM_ADVISORS)]

print("\n🔮 Running predictions across all advisors...")
with torch.no_grad():
    for batch_idx, (x_cnn, x_tab) in enumerate(tqdm(loader, desc="Evaluating grids", ncols=90)):
        x_cnn, x_tab = x_cnn.to(DEVICE), x_tab.to(DEVICE)
        batch_preds = []
        for model in models:
            preds = model(x_cnn, x_tab)
            batch_preds.append(preds.cpu())

        # Concatenate batch predictions per advisor
        for i in range(NUM_ADVISORS):
            advisor_preds[i] = torch.cat((advisor_preds[i], batch_preds[i]), dim=0)

        # Print counter feedback
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * BATCH_SIZE} grids...")

advisor_preds_np = np.hstack([p.numpy() for p in advisor_preds])
print(f"✅ Prediction complete — shape: {advisor_preds_np.shape}")

# --------------------------------------------------------------------------
# 5. SELECT HIGH-SCORING GRIDS
# --------------------------------------------------------------------------
mask = np.all(advisor_preds_np >= THRESHOLD, axis=1)
selected_indices = np.where(mask)[0]
selected_grids = new_grids[selected_indices]
selected_scores = advisor_preds_np[selected_indices]

print(f"✅ {len(selected_grids)} of {len(new_grids)} grids meet the threshold (≥ {THRESHOLD})")

# --------------------------------------------------------------------------
# 6. SAVE APPROVED GRIDS
# --------------------------------------------------------------------------
approved_path = os.path.join(OUTPUT_DIR, "approved_grids.npy")
scores_path = os.path.join(OUTPUT_DIR, "approved_scores.npy")

np.save(approved_path, selected_grids)
np.save(scores_path, selected_scores)

print(f"💾 Saved approved grids → {approved_path}")
print(f"💾 Saved their predicted scores → {scores_path}")
