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

# 1. Dataset Loading (MNIST subset for CPU speed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset of 10,000 training images and 2,000 test images
train_subset_indices = list(range(10000))
test_subset_indices = list(range(2000))

train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_subset_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

# Helper function to add Gaussian noise
def add_noise(img, noise_factor=0.5):
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img

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

print("\n--- Phase 1: Pre-training Denoising Autoencoder ---")
dae = DAE().to(device)
dae_criterion = nn.MSELoss()
dae_optimizer = optim.Adam(dae.parameters(), lr=0.002)

for epoch in range(1, 6):
    dae.train()
    running_loss = 0.0
    for data, _ in train_loader:
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
print("Memory Bank populated successfully. Shape:", prototype_tensor.shape)

# 4. Define Fixed Memory-Augmented Classifier
class FixedMemoryClassifier(nn.Module):
    def __init__(self, encoder, memory_bank, global_mean, mode='cosine_centered'):
        super(FixedMemoryClassifier, self).__init__()
        self.encoder = encoder
        self.mode = mode
        self.register_buffer('memory_bank', memory_bank)
        self.register_buffer('global_mean', global_mean)
        
        # Downstream classification head
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        # Extract features using frozen encoder
        z = self.encoder(x.view(-1, 28 * 28))
        
        if self.mode == 'baseline':
            z_combined = z
        elif self.mode == 'cosine_raw':
            # Previous flawed raw cosine similarity
            z_norm = F.normalize(z, p=2, dim=1)
            mem_norm = F.normalize(self.memory_bank, p=2, dim=1)
            similarity = torch.matmul(z_norm, mem_norm.t())
            attention_weights = F.softmax(similarity / 0.2, dim=1)
            z_retrieved = torch.matmul(attention_weights, self.memory_bank)
            z_combined = z + z_retrieved
        elif self.mode == 'cosine_centered':
            # Option 1: Mean Centering
            z_centered = z - self.global_mean.unsqueeze(0)
            mem_centered = self.memory_bank - self.global_mean.unsqueeze(0)
            
            z_norm = F.normalize(z_centered, p=2, dim=1)
            mem_norm = F.normalize(mem_centered, p=2, dim=1)
            similarity = torch.matmul(z_norm, mem_norm.t())
            attention_weights = F.softmax(similarity / 0.2, dim=1)
            z_retrieved = torch.matmul(attention_weights, self.memory_bank)
            z_combined = z + z_retrieved
        elif self.mode == 'euclidean':
            # Option 2: Negative Euclidean Distance
            # z shape: (batch, 32), memory_bank shape: (10, 32)
            dist = torch.cdist(z, self.memory_bank)  # (batch, 10)
            # Use negative distance with temperature scale of 2.0
            attention_weights = F.softmax(-dist / 2.0, dim=1)
            z_retrieved = torch.matmul(attention_weights, self.memory_bank)
            z_combined = z + z_retrieved
            
        return self.fc(z_combined)

# 5. Train Downstream Classifiers
def train_classifier(mode, epochs=10):
    print(f"\n--- Training Classifier: {mode} ---")
    model = FixedMemoryClassifier(dae.encoder, prototype_tensor, global_mean_tensor, mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.002)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            noisy_data = add_noise(data, 0.5)
            noisy_data, target = noisy_data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_classifier(model, criterion)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    return history

def evaluate_classifier(model, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            noisy_data = add_noise(data, 0.5)
            noisy_data, target = noisy_data.to(device), target.to(device)
            output = model(noisy_data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    val_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)
    return val_loss, val_acc

# Run all configurations
modes = ['baseline', 'cosine_raw', 'cosine_centered', 'euclidean']
results = {}

for mode in modes:
    results[mode] = train_classifier(mode, epochs=10)

# 6. Plot results
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
for mode in modes:
    plt.plot(results[mode]['val_acc'], label=mode, marker='o')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
for mode in modes:
    plt.plot(results[mode]['val_loss'], label=mode, marker='s')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fixed_prototype_results.png')
print("\nSaved plot to fixed_prototype_results.png")

# Print Summary Table
print("\nFinal Downstream Classifier Summary (after 10 epochs):")
print("-" * 50)
print(f"{'Configuration':<20} | {'Val Loss':<10} | {'Val Acc (%)':<12}")
print("-" * 50)
for mode in modes:
    final_loss = results[mode]['val_loss'][-1]
    final_acc = results[mode]['val_acc'][-1]
    print(f"{mode:<20} | {final_loss:<10.4f} | {final_acc:<12.2f}")
print("-" * 50)
