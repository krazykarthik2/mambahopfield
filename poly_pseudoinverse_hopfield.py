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

# Compute global mean of memories for centering
global_mean = torch.mean(X, dim=1, keepdim=True) # (784, 1)

# Center memories
X_centered = X - global_mean # (784, 10)

# Define polynomial feature map
p = 3
def phi_poly(x):
    return x ** p

# Compute Centered Polynomial features for memories
phi_X = phi_poly(X_centered) # (784, 10)

# Compute Pseudoinverse of the polynomial centered features
pinv_phi = torch.pinverse(phi_X) # (10, 784)

# Pre-compute the 784x784 Centered Polynomial Pseudoinverse Memory Matrix S
S_matrix = torch.matmul(X, pinv_phi) # (784, 784)

# 2. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0 # Zero out bottom 50%

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 3. Retrieval Update Loop using the pre-computed S_matrix
def retrieve_poly_pinv(query, S, global_mean, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        # Center the state
        state_centered = state - global_mean.squeeze(1)
        # Apply polynomial feature map
        phi_state = phi_poly(state_centered)
        # Apply the precomputed O(d^2) matrix multiplication
        state = torch.matmul(S, phi_state)
        # Clamp to valid pixel range
        state = torch.clamp(state, 0.0, 1.0)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
traj_erased = retrieve_poly_pinv(erased_query, S_matrix, global_mean, steps=steps)
traj_noisy = retrieve_poly_pinv(noisy_query, S_matrix, global_mean, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
mse_erased = F.mse_loss(traj_erased[-1], clean_target).item()
mse_noisy = F.mse_loss(traj_noisy[-1], clean_target_noise).item()

print("\nCentered Polynomial Pseudoinverse Reconstruction MSE Summary:")
print("-" * 65)
print(f"Erased Query final MSE: {mse_erased:.8f}")
print(f"Noisy Query final MSE:  {mse_noisy:.8f}")
print("-" * 65)

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
plt.savefig('poly_pseudoinverse_results.png')
print("\nSaved visual trajectory plot to poly_pseudoinverse_results.png")
