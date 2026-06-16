import os
import torch
import torch.nn as nn
import torch.optim as optim
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

# 1. Dataset Loading (MNIST with subset for fast training on CPU)
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset of 10,000 training images and 1,000 test images for speed
train_subset_indices = list(range(10000))
test_subset_indices = list(range(1000))

train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_subset_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# Helper function to add Gaussian noise
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

# 2. Denoising Autoencoder Model Definition

class DenoisingAutoencoder(nn.Module):
    def __init__(self, mode='trainable_pre'):
        super(DenoisingAutoencoder, self).__init__()
        self.mode = mode
        
        # Encoder: 784 -> 256 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )
        
        # Pre-layers ("Memory Bank" injected at the bottleneck)
        if mode in ['trainable_pre', 'frozen_pre']:
            self.pre_layers = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
            # Constant input to memory bank
            self.register_buffer('constant_input', torch.ones(1, 32))
            
            if mode == 'frozen_pre':
                for param in self.pre_layers.parameters():
                    param.requires_grad = False
                    
        elif mode == 'bias_param':
            self.bias_param = nn.Parameter(torch.zeros(32))
            
        # Decoder: 32 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is the noisy image
        x_flat = x.view(-1, 28 * 28)
        z = self.encoder(x_flat)
        
        if self.mode in ['trainable_pre', 'frozen_pre']:
            const = self.constant_input.expand(x.size(0), -1)
            z_mem = self.pre_layers(const)
            z_combined = z + z_mem
        elif self.mode == 'bias_param':
            z_combined = z + self.bias_param.unsqueeze(0)
        else: # baseline (no memory bank)
            z_combined = z
            
        reconstruction = self.decoder(z_combined)
        return reconstruction.view(-1, 1, 28, 28)

# 3. Training and Evaluation Functions

def train_dae(mode, epochs=5):
    print(f"\n--- Training DAE: {mode} ---")
    model = DenoisingAutoencoder(mode=mode).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, _ in train_loader:
            # Add noise to input images
            noisy_data = add_noise(data, noise_factor=0.5)
            
            noisy_data, clean_data = noisy_data.to(device), data.to(device)
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, clean_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_dae(model, criterion)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f}")
        
    return model, history

def evaluate_dae(model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            noisy_data = add_noise(data, noise_factor=0.5)
            noisy_data, clean_data = noisy_data.to(device), data.to(device)
            output = model(noisy_data)
            val_loss += criterion(output, clean_data).item() * data.size(0)
    return val_loss / len(test_loader.dataset)

# 4. Run Experiment
modes = ['baseline', 'frozen_pre', 'trainable_pre', 'bias_param']
results = {}
trained_models = {}

for mode in modes:
    model, history = train_dae(mode, epochs=5)
    results[mode] = history
    trained_models[mode] = model

# 5. Save Visual Reconstruction Comparison
print("\nGenerating reconstruction visual comparison...")
# Pick 5 random clean images from the test loader
test_iter = iter(test_loader)
clean_samples, _ = next(test_iter)
clean_samples = clean_samples[:5]
noisy_samples = add_noise(clean_samples, noise_factor=0.5)

fig, axes = plt.subplots(6, 5, figsize=(10, 12))

# Row 0: Original Clean Images
for i in range(5):
    axes[0, i].imshow(clean_samples[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("Original Clean", loc='left')

# Row 1: Noisy Inputs
for i in range(5):
    axes[1, i].imshow(noisy_samples[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title("Noisy Input", loc='left')

# Rows 2-5: Reconstructions from different models
for row_idx, mode in enumerate(modes):
    model = trained_models[mode]
    model.eval()
    with torch.no_grad():
        reconstructed = model(noisy_samples.to(device)).cpu()
    for i in range(5):
        axes[row_idx + 2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[row_idx + 2, i].axis('off')
        if i == 0:
            axes[row_idx + 2, i].set_title(f"Recon: {mode}", loc='left')

plt.tight_layout()
plt.savefig('dae_reconstructions.png')
print("Saved visual reconstruction comparison to dae_reconstructions.png")

# 6. Plot Loss Curve Comparison
plt.figure(figsize=(8, 5))
for mode in modes:
    plt.plot(results[mode]['val_loss'], label=mode, marker='o')
plt.title('Validation Reconstruction Loss (MSE) Comparison')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('dae_loss_curves.png')
print("Saved loss curves to dae_loss_curves.png")

# Summary Table
print("\nFinal DAE Results Summary (after 5 epochs):")
print("-" * 50)
print(f"{'Configuration':<20} | {'Val MSE Loss':<15}")
print("-" * 50)
for mode in modes:
    final_loss = results[mode]['val_loss'][-1]
    print(f"{mode:<20} | {final_loss:<15.4f}")
print("-" * 50)
