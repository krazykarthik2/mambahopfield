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

# 2. Pre-compute Mamba/Linear Hopfield Memory matrix S and norm vector
# Feature map: phi(x) = elu(x) + 1
def phi(x):
    return F.elu(x) + 1.0

phi_X = phi(X) # (784, 10)
S = torch.matmul(X, phi_X.t()) # (784, 784) - Fixed size memory matrix!
z_norm = torch.sum(phi_X, dim=1) # (784,)

# 3. Create corrupted queries
# Query 1: Erased Bottom Half of memory index 4 (Coat)
clean_target = X[:, 4].clone()
erased_query = clean_target.clone()
erased_query[392:] = 0.0 # Zero out the bottom 50% of the pixels

# Query 2: Heavy Noise added to memory index 7 (Sneaker)
clean_target_noise = X[:, 7].clone()
noisy_query = clean_target_noise + 0.6 * torch.randn_like(clean_target_noise)
noisy_query = torch.clamp(noisy_query, 0., 1.)

# 4. Hopfield Retrieval Update Loops
def retrieve_softmax(query, X, beta=40.0, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        # dot_products = X^T * state
        dot_products = torch.matmul(X.t(), state)
        weights = F.softmax(beta * dot_products, dim=0)
        # state = X * weights
        state = torch.matmul(X, weights)
        trajectory.append(state.clone())
    return trajectory

def retrieve_linear(query, S, z_norm, steps=5):
    state = query.clone()
    trajectory = [state.clone()]
    for _ in range(steps):
        phi_state = phi(state)
        # numerator = S * phi(state)
        numerator = torch.matmul(S, phi_state)
        # denominator = z_norm * phi(state)
        denominator = torch.dot(z_norm, phi_state)
        state = numerator / (denominator + 1e-6)
        trajectory.append(state.clone())
    return trajectory

# Run retrievals
steps = 5
softmax_traj_erased = retrieve_softmax(erased_query, X, beta=40.0, steps=steps)
linear_traj_erased = retrieve_linear(erased_query, S, z_norm, steps=steps)

softmax_traj_noisy = retrieve_softmax(noisy_query, X, beta=40.0, steps=steps)
linear_traj_noisy = retrieve_linear(noisy_query, S, z_norm, steps=steps)

# Calculate final pixel-to-pixel reconstruction error (MSE)
print("\nReconstruction MSE Comparison:")
print("-" * 60)
print(f"Erased Query (Softmax Hopfield) final MSE: {F.mse_loss(softmax_traj_erased[-1], clean_target).item():.6f}")
print(f"Erased Query (Linear Hopfield) final MSE:  {F.mse_loss(linear_traj_erased[-1], clean_target).item():.6f}")
print(f"Noisy Query (Softmax Hopfield) final MSE:  {F.mse_loss(softmax_traj_noisy[-1], clean_target_noise).item():.6f}")
print(f"Noisy Query (Linear Hopfield) final MSE:   {F.mse_loss(linear_traj_noisy[-1], clean_target_noise).item():.6f}")
print("-" * 60)

# 5. Visual Trajectory Comparison
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Plot columns correspond to step t = 0, 1, 2, 3, 4, 5
cols = [0, 1, 2, 3, 4, 5]

for col_idx, t in enumerate(cols):
    # Row 0: Softmax Hopfield (Erased Input)
    axes[0, col_idx].imshow(softmax_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[0, col_idx].axis('off')
    axes[0, col_idx].set_title(f"t = {t}")
    if col_idx == 0:
        axes[0, col_idx].set_ylabel("Softmax (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: Linear Hopfield (Erased Input)
    axes[1, col_idx].imshow(linear_traj_erased[t].view(28, 28).numpy(), cmap='gray')
    axes[1, col_idx].axis('off')
    if col_idx == 0:
        axes[1, col_idx].set_ylabel("Linear (Erased)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

    # Row 2: Softmax Hopfield (Noisy Input)
    axes[2, col_idx].imshow(softmax_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[2, col_idx].axis('off')
    if col_idx == 0:
        axes[2, col_idx].set_ylabel("Softmax (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 3: Linear Hopfield (Noisy Input)
    axes[3, col_idx].imshow(linear_traj_noisy[t].view(28, 28).numpy(), cmap='gray')
    axes[3, col_idx].axis('off')
    if col_idx == 0:
        axes[3, col_idx].set_ylabel("Linear (Noisy)", labelpad=25, rotation=0, ha='right', va='center', fontweight='bold')

# Enable labels
for row in range(4):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('mhn_results.png')
print("\nSaved visual trajectory plot to mhn_results.png")
