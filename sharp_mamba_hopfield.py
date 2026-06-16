import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 1. Select 10 distinct memory patterns (one from each clothing class)
memories = []
classes_found = set()
for img, label in dataset:
    if label not in classes_found:
        memories.append(img.view(-1))  # Flatten to size 784
        classes_found.add(label)
        if len(classes_found) == 10:
            break

X = torch.stack(memories, dim=1) # (784, 10)
print("Memory Matrix X shape:", X.shape)

# Define the low-rank projection weight matrix
W_proj = X.t() # (10, 784)

# 2. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0 # Zero out the bottom 50%

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 3. Dual-Projection Hopfield Update Loop
def retrieve_dual_projection(query, X, W_proj, beta=40.0, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        # Step A: Project to memory dimension - O(d * M) complexity
        memory_scores = torch.matmul(W_proj, state) # Shape: (10,)
        
        # Step B: Sharp selection (Softmax)
        weights = F.softmax(beta * memory_scores, dim=0) # Shape: (10,)
        
        # Step C: Project back to pixel space - O(d * M) complexity
        state = torch.matmul(X, weights) # Shape: (784,)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
traj_erased = retrieve_dual_projection(erased_query, X, W_proj, beta=40.0, steps=steps)
traj_noisy = retrieve_dual_projection(noisy_query, X, W_proj, beta=40.0, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
mse_erased = F.mse_loss(traj_erased[-1], clean_target).item()
mse_noisy = F.mse_loss(traj_noisy[-1], clean_target_noise).item()

print("\nDual-Projection Reconstruction MSE Summary:")
print("-" * 50)
print(f"Erased Query final MSE: {mse_erased:.8f}")
print(f"Noisy Query final MSE:  {mse_noisy:.8f}")
print("-" * 50)

# 4. Visual Trajectory Comparison
fig, axes = plt.subplots(2, 6, figsize=(12, 5))
cols = [0, 1, 2, 3, 4, 5]

for col_idx, t in enumerate(cols):
    # Row 0: Erased Input Trajectory
    axes[0, col_idx].imshow(traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"t = {t}")
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Erased Traj", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: Noisy Input Trajectory
    axes[1, col_idx].imshow(traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[1, col_idx].axis('off')
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("Noisy Traj", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

# Enable labels
for row in range(2):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('sharp_mamba_results.png')
print("\nSaved visual trajectory plot to sharp_mamba_results.png")
