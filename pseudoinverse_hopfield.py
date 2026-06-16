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

# Target memory matrix X: (784, 10)
X = torch.stack(memories, dim=1) 
print("Memory Matrix X shape:", X.shape)

# Compute global mean
global_mean = torch.mean(X, dim=1, keepdim=True) # (784, 1)
# Centered memories
X_centered = X - global_mean # (784, 10)

# Compute Pseudoinverse of centered memories
# pinv shape: (10, 784)
pinv = torch.pinverse(X_centered)

# 2. Pre-compute the 784x784 Pseudoinverse Memory Matrix S
S_matrix = torch.matmul(X, pinv) # (784, 784) - Fixed size matrix!

# 3. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 4. Retrieval Update Loops
def retrieve_pure_linear_pinv(query, S, global_mean, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        state_centered = state - global_mean.squeeze(1)
        # O(d^2) matrix-vector product
        state = torch.matmul(S, state_centered)
        state = torch.clamp(state, 0.0, 1.0)
        trajectory.append(state.clone())
    return trajectory

def retrieve_pseudoinverse_sharp(query, X, pinv, global_mean, beta=40.0, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        # Center the query
        state_centered = state - global_mean.squeeze(1)
        # Step A: Project to coordinates using pseudoinverse
        coords = torch.matmul(pinv, state_centered) # (10,)
        # Step B: Sharp selection (Softmax)
        weights = F.softmax(beta * coords, dim=0) # (10,)
        # Step C: Reconstruct clean image
        state = torch.matmul(X, weights) # (784,)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
pure_traj_erased = retrieve_pure_linear_pinv(erased_query, S_matrix, global_mean, steps=steps)
sharp_traj_erased = retrieve_pseudoinverse_sharp(erased_query, X, pinv, global_mean, beta=45.0, steps=steps)

pure_traj_noisy = retrieve_pure_linear_pinv(noisy_query, S_matrix, global_mean, steps=steps)
sharp_traj_noisy = retrieve_pseudoinverse_sharp(noisy_query, X, pinv, global_mean, beta=45.0, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
print("\nPseudoinverse Reconstruction MSE Comparison:")
print("-" * 65)
print(f"Erased Query (Pure Linear S) final MSE:       {F.mse_loss(pure_traj_erased[-1], clean_target).item():.8f}")
print(f"Erased Query (Pseudoinverse Sharp) final MSE: {F.mse_loss(sharp_traj_erased[-1], clean_target).item():.8f}")
print("-" * 65)
print(f"Noisy Query (Pure Linear S) final MSE:        {F.mse_loss(pure_traj_noisy[-1], clean_target_noise).item():.8f}")
print(f"Noisy Query (Pseudoinverse Sharp) final MSE:  {F.mse_loss(sharp_traj_noisy[-1], clean_target_noise).item():.8f}")
print("-" * 65)

# 5. Visual Trajectory Comparison
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
cols = [0, 1, 2, 3, 4, 5]

for col_idx, t in enumerate(cols):
    # --- ERASED QUERIES ---
    # Row 0: Pure Linear Matrix S
    axes[0, col_idx].imshow(pure_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"t = {t}")
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Linear S (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: Pseudoinverse Sharp
    axes[1, col_idx].imshow(sharp_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[1, col_idx].axis('off')
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("Sharp Pinv (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # --- NOISY QUERIES ---
    # Row 2: Pure Linear Matrix S
    axes[2, col_idx].imshow(pure_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[2, col_idx].axis('off')
    if col_idx == 0:
        axes[2, col_idx].set_ylabel("Linear S (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 3: Pseudoinverse Sharp
    axes[3, col_idx].imshow(sharp_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[3, col_idx].axis('off')
    if col_idx == 0:
        axes[3, col_idx].set_ylabel("Sharp Pinv (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

# Enable labels
for row in range(4):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('pseudoinverse_results.png')
print("\nSaved visual trajectory plot to pseudoinverse_results.png")
