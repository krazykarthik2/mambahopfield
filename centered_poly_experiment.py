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

# 2. Pre-compute Centered Polynomial Linear Hopfield (Power p=7)
# Feature map: phi(x) = x ** 7 (element-wise)
# An odd power preserves sign, enabling negative activations
p = 7
def phi_poly(x):
    return x ** p

phi_X_poly = phi_poly(X_centered) # (784, 10)
S_poly = torch.matmul(X_centered, phi_X_poly.t()) # (784, 784)
z_norm_poly = torch.sum(phi_X_poly, dim=1) # (784,)

# 3. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0 # Zero out the bottom 50%

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 4. Retrieval Update Loops
def retrieve_softmax(query, X, beta=40.0, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        dot_products = torch.matmul(X.t(), state)
        weights = F.softmax(beta * dot_products, dim=0)
        state = torch.matmul(X, weights)
        trajectory.append(state.clone())
    return trajectory

def retrieve_linear_centered_poly(query, S, z_norm, global_mean, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        # Center the state
        state_centered = state - global_mean.squeeze(1)
        phi_state = phi_poly(state_centered)
        
        # S * phi(state)
        numerator = torch.matmul(S, phi_state)
        # z_norm * phi(state)
        denominator = torch.dot(z_norm, phi_state)
        
        # Update rule
        state_centered = numerator / (denominator + 1e-6)
        
        # Add mean back
        state = state_centered + global_mean.squeeze(1)
        # Clamp to valid pixel range [0, 1]
        state = torch.clamp(state, 0.0, 1.0)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
softmax_traj_erased = retrieve_softmax(erased_query, X, beta=40.0, steps=steps)
poly_traj_erased = retrieve_linear_centered_poly(erased_query, S_poly, z_norm_poly, global_mean, steps=steps)

softmax_traj_noisy = retrieve_softmax(noisy_query, X, beta=40.0, steps=steps)
poly_traj_noisy = retrieve_linear_centered_poly(noisy_query, S_poly, z_norm_poly, global_mean, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
print("\nReconstruction MSE Comparison:")
print("-" * 65)
print(f"Erased Query (Softmax Hopfield) final MSE: {F.mse_loss(softmax_traj_erased[-1], clean_target).item():.6f}")
print(f"Erased Query (Centered Poly Linear Hopfield) final MSE: {F.mse_loss(poly_traj_erased[-1], clean_target).item():.6f}")
print("-" * 65)
print(f"Noisy Query (Softmax Hopfield) final MSE:  {F.mse_loss(softmax_traj_noisy[-1], clean_target_noise).item():.6f}")
print(f"Noisy Query (Centered Poly Linear Hopfield) final MSE:  {F.mse_loss(poly_traj_noisy[-1], clean_target_noise).item():.6f}")
print("-" * 65)

# 5. Visual Trajectory Comparison
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
cols = [0, 1, 2, 3, 4, 5]

for col_idx, t in enumerate(cols):
    # --- ERASED QUERIES ---
    # Row 0: Softmax Hopfield
    axes[0, col_idx].imshow(softmax_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"t = {t}")
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Softmax (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: Centered Poly Linear Hopfield
    axes[1, col_idx].imshow(poly_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[1, col_idx].axis('off')
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("Poly Linear (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # --- NOISY QUERIES ---
    # Row 2: Softmax Hopfield
    axes[2, col_idx].imshow(softmax_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[2, col_idx].axis('off')
    if col_idx == 0:
        axes[2, col_idx].set_ylabel("Softmax (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 3: Centered Poly Linear Hopfield
    axes[3, col_idx].imshow(poly_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[3, col_idx].axis('off')
    if col_idx == 0:
        axes[3, col_idx].set_ylabel("Poly Linear (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

# Enable labels
for row in range(4):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('centered_poly_results.png')
print("\nSaved visual trajectory plot to centered_poly_results.png")
