import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Dataset Loading (FashionMNIST subset)
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset of 10,000 training images and 20 test images for final demonstration
train_subset_indices = list(range(10000))
test_subset_indices = list(range(20))

train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_subset_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)

# Helper function to add Gaussian noise (used for training the DAE)
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

# Helper function to apply HEAVY ERASURE (e.g. zero out the bottom half of the image)
def erase_bottom_half(img):
    erased_img = img.clone()
    # img shape: (batch, 1, 28, 28)
    erased_img[:, :, 14:, :] = 0.0
    return erased_img

# 2. Denoising Autoencoder Definition (Phase 1)
class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x.view(-1, 28*28))
        recon = self.decoder(z)
        return recon.view(-1, 1, 28, 28)

print("\n--- Phase 1: Pre-training Denoising Autoencoder on FashionMNIST ---")
dae = DAE().to(device)
dae_criterion = nn.MSELoss()
dae_optimizer = optim.Adam(dae.parameters(), lr=0.002)

for epoch in range(1, 6):
    dae.train()
    running_loss = 0.0
    for data, _ in train_loader:
        # Train DAE with standard noise so it learns to clean/restore
        noisy_data = add_noise(data, 0.5).to(device)
        clean_data = data.to(device)
        
        dae_optimizer.zero_grad()
        output = dae(noisy_data)
        loss = dae_criterion(output, clean_data)
        loss.backward()
        dae_optimizer.step()
        running_loss += loss.item() * data.size(0)
    print(f"DAE Epoch {epoch}/5 - Reconstruction Loss: {running_loss/len(train_loader.dataset):.4f}")

# Freeze DAE
for param in dae.parameters():
    param.requires_grad = False
dae.eval()

# 3. Populate Memory Bank (10 class prototypes)
print("\n--- Populating Latent Prototype Memory Bank ---")
latent_vectors = [[] for _ in range(10)]
all_vectors = []

with torch.no_grad():
    for data, targets in train_loader:
        data = data.to(device)
        z = dae.encoder(data.view(-1, 28*28))
        all_vectors.append(z.cpu().numpy())
        for zi, target in zip(z, targets):
            latent_vectors[target.item()].append(zi.cpu().numpy())

# Compute global mean of all latent vectors (for centering)
global_mean = np.mean(np.concatenate(all_vectors, axis=0), axis=0)
global_mean_tensor = torch.tensor(global_mean, dtype=torch.float32).to(device)

# Compute average latent vector for each class
prototypes = []
for i in range(10):
    class_mean = np.mean(latent_vectors[i], axis=0)
    prototypes.append(class_mean)
    
prototype_tensor = torch.tensor(np.array(prototypes), dtype=torch.float32).to(device)

# Pre-compute Centered Memory matrix and norm vector
phi_K = F.elu(prototype_tensor - global_mean_tensor) + 1.0       # (10, 32)
phi_V = prototype_tensor - global_mean_tensor                     # (10, 32)
S_matrix_centered = torch.matmul(phi_K.t(), phi_V)               # (32, 32)
z_norm_centered = torch.sum(phi_K, dim=0)                         # (32,)

# 4. Image Reconstruction Evaluation on Heavily Erased Inputs
print("\n--- Running Erasure Reconstruction Test ---")
dae.eval()

# Get some test samples
test_iter = iter(test_loader)
clean_samples, targets = next(test_iter)

# Pick 5 distinct samples from different classes
selected_indices = [0, 1, 2, 4, 8]  # Corresponds to a variety of clothing items
clean_samples = clean_samples[selected_indices]
targets = targets[selected_indices]

# Apply heavy erasure (bottom 50% zeroed out)
erased_samples = erase_bottom_half(clean_samples)

# Perform reconstructions
with torch.no_grad():
    # A. Baseline Reconstruction
    z_baseline = dae.encoder(erased_samples.view(-1, 28*28).to(device))
    recon_baseline = dae.decoder(z_baseline).cpu()
    
    # B. Memory-Augmented (Centered Linear Hopfield) Reconstruction
    z_erased = dae.encoder(erased_samples.view(-1, 28*28).to(device))
    z_centered = z_erased - global_mean_tensor.unsqueeze(0)
    phi_z = F.elu(z_centered) + 1.0
    
    # Retrieve Centered Offset
    numerator = torch.matmul(phi_z, S_matrix_centered)
    denominator = torch.matmul(phi_z, z_norm_centered.unsqueeze(1))
    z_retrieved = numerator / (denominator + 1e-6)
    
    # Combine representation and reconstruct
    z_combined = z_erased + z_retrieved
    recon_memory = dae.decoder(z_combined).cpu()

# 5. Visualize Results
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

for i in range(5):
    # Row 0: Original Clean Image
    axes[0, i].imshow(clean_samples[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(classes[targets[i].item()], fontsize=10)
    if i == 0:
        axes[0, i].set_ylabel("Original Clean", labelpad=20, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 1: Heavily Erased Image
    axes[1, i].imshow(erased_samples[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel("Erased Input", labelpad=20, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 2: Baseline Reconstruction
    axes[2, i].imshow(recon_baseline[i].view(28, 28).squeeze(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_ylabel("DAE Baseline", labelpad=20, rotation=0, ha='right', va='center', fontweight='bold')
        
    # Row 3: Memory-Augmented Reconstruction
    axes[3, i].imshow(recon_memory[i].view(28, 28).squeeze(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0:
        axes[3, i].set_ylabel("Linear Hopfield", labelpad=20, rotation=0, ha='right', va='center', fontweight='bold')

# Re-enable ylabels visibility (since axis('off') hides them)
for row in range(4):
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('erased_reconstructions.png')
print("Saved visualization to erased_reconstructions.png")
