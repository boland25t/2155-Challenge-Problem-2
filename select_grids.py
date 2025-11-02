import numpy as np
import matplotlib.pyplot as plt
import os

# === Parameters ===
grid_path = "selected_grids/approved_grids.npy"
score_path = "selected_grids/approved_scores.npy"
output_grid_path = "selected_grids/sample_100_grids.npy"
output_score_path = "selected_grids/sample_100_scores.npy"
n_select = 100

# === Load ===
grids = np.load(grid_path)
scores = np.load(score_path)
print(f"Loaded {len(grids)} grids and {scores.shape[1]} advisor scores each.")

# === Select 100 random grids ===
np.random.seed(42)
idx = np.random.choice(len(grids), size=n_select, replace=False)
selected_grids = grids[idx]
selected_scores = scores[idx]

# === Save new batch ===
os.makedirs("selected_grids", exist_ok=True)
np.save(output_grid_path, selected_grids)
np.save(output_score_path, selected_scores)
print(f"✅ Saved {n_select} random grids and scores to 'selected_grids/'")

# === Visualization ===
cols = 10
rows = n_select // cols
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
fig.suptitle("Random 100 Approved Grids with Advisor Scores", fontsize=18)

for i, ax in enumerate(axes.flat):
    g = selected_grids[i]
    sc = selected_scores[i]
    ax.imshow(g, cmap="viridis", vmin=0, vmax=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"A0:{sc[0]:.2f}\nA1:{sc[1]:.2f}\nA2:{sc[2]:.2f}\nA3:{sc[3]:.2f}",
        fontsize=8,
    )

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
