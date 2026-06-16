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

# Shape: (784, 10) - 10 stored clean memories
X = torch.stack(memories, dim=1) # (784, 10)
print("Memory Matrix X shape:", X.shape)

# 2. Pre-compute standard ELU Linear Hopfield
def phi_elu(x):
    return F.elu(x) + 1.0

phi_X_elu = phi_elu(X)
S_elu = torch.matmul(X, phi_X_elu.t())
z_norm_elu = torch.sum(phi_X_elu, dim=1)

# 3. Pre-compute Polynomial Linear Hopfield (Power p=8)
# Feature map: phi(x) = x ** 8 (element-wise)
p = 8
def phi_poly(x):
    return x ** p

phi_X_poly = phi_poly(X)
S_poly = torch.matmul(X, phi_X_poly.t()) # (784, 784) - Fixed size matrix!
z_norm_poly = torch.sum(phi_X_poly, dim=1) # (784,)

# 4. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0 # Zero out the bottom 50% of the pixels

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 5. Retrieval Update Loops
def retrieve_softmax(query, X, beta=40.0, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        dot_products = torch.matmul(X.t(), state)
        weights = F.softmax(beta * dot_products, dim=0)
        state = torch.matmul(X, weights)
        trajectory.append(state.clone())
    return trajectory

def retrieve_linear(query, S, z_norm, phi_fn, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        phi_state = phi_fn(state)
        numerator = torch.matmul(S, phi_state)
        denominator = torch.dot(z_norm, phi_state)
        state = numerator / (denominator + 1e-6)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
softmax_traj_erased = retrieve_softmax(erased_query, X, beta=40.0, steps=steps)
elu_traj_erased = retrieve_linear(erased_query, S_elu, z_norm_elu, phi_elu, steps=steps)
poly_traj_erased = retrieve_linear(erased_query, S_poly, z_norm_poly, phi_poly, steps=steps)

softmax_traj_noisy = retrieve_softmax(noisy_query, X, beta=40.0, steps=steps)
elu_traj_noisy = retrieve_linear(noisy_query, S_elu, z_norm_elu, phi_elu, steps=steps)
poly_traj_noisy = retrieve_linear(noisy_query, S_poly, z_norm_poly, phi_poly, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
print("\nReconstruction MSE Comparison:")
print("-" * 65)
print(f"Erased Query (Softmax Hopfield) final MSE: {F.mse_loss(softmax_traj_erased[-1], clean_target).item():.6f}")
print(f"Erased Query (ELU Linear Hopfield) final MSE: {F.mse_loss(elu_traj_erased[-1], clean_target).item():.6f}")
print(f"Erased Query (Poly Linear Hopfield) final MSE: {F.mse_loss(poly_traj_erased[-1], clean_target).item():.6f}")
print("-" * 65)
print(f"Noisy Query (Softmax Hopfield) final MSE:  {F.mse_loss(softmax_traj_noisy[-1], clean_target_noise).item():.6f}")
print(f"Noisy Query (ELU Linear Hopfield) final MSE:  {F.mse_loss(elu_traj_noisy[-1], clean_target_noise).item():.6f}")
print(f"Noisy Query (Poly Linear Hopfield) final MSE:  {F.mse_loss(poly_traj_noisy[-1], clean_target_noise).item():.6f}")
print("-" * 65)

# 6. Visual Trajectory Comparison
fig, axes = plt.subplots(6, 6, figsize=(12, 12))

# Plot columns correspond to step t = 0, 1, 2, 3, 4, 5
cols = [0, 1, 2, 3, 4, 5]

for col_idx, t in enumerate(cols):
    # --- ERASED QUERIES ---
    # Row 0: Softmax Hopfield
    axes[0, col_idx].imshow(softmax_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"t = {t}")
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Softmax (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: ELU Linear Hopfield (previous flawed)
    axes[1, col_idx].imshow(elu_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[1, col_idx].axis('off')
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("ELU Linear (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # Row 2: Polynomial Linear Hopfield (new)
    axes[2, col_idx].imshow(poly_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[2, col_idx].axis('off')
    if col_idx == 0:
        axes[2, col_idx].set_ylabel("Poly Linear (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # --- NOISY QUERIES ---
    # Row 3: Softmax Hopfield
    axes[3, col_idx].imshow(softmax_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[3, col_idx].axis('off')
    if col_idx == 0:
        axes[3, col_idx].set_ylabel("Softmax (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 4: ELU Linear Hopfield
    axes[4, col_idx].imshow(elu_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[4, col_idx].axis('off')
    if col_idx == 0:
        axes[4, col_idx].set_ylabel("ELU Linear (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # Row 5: Polynomial Linear Hopfield
    axes[5, col_idx].imshow(poly_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[5, col_idx].axis('off')
    if col_idx == 0:
        axes[5, col_idx].set_ylabel("Poly Linear (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

# Enable labels
for row in range(6):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('sharp_hopfield_results.png')
print("\nSaved visual trajectory plot to sharp_hopfield_results.png")
