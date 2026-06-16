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

# 1. Dataset Loading (MNIST - using subset for speed on CPU)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset of 10,000 training images and 2,000 test images for speed
train_subset_indices = list(range(10000))
test_subset_indices = list(range(2000))

train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_subset_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

# 2. Model Definition

class GaussianMiddleInputNet(nn.Module):
    def __init__(self, mode='ones'):
        super(GaussianMiddleInputNet, self).__init__()
        self.mode = mode
        
        # Input projection layer
        self.input_proj = nn.Linear(28 * 28, 128)
        
        # Pre-layers
        self.pre_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        if mode == 'ones':
            # Constant input to pre-layers: all ones
            self.register_buffer('start_node_input', torch.ones(1, 128))
        elif mode == 'fixed_gaussian':
            # Constant input to pre-layers: a fixed random Gaussian vector
            self.register_buffer('start_node_input', torch.randn(1, 128))
        # Note: 'sample_gaussian' mode generates the vector dynamically during the forward pass
            
        # Post-layers
        self.post_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        h_in = self.input_proj(x)
        
        if self.mode in ['ones', 'fixed_gaussian']:
            const = self.start_node_input.expand(x.size(0), -1)
            h_pre = self.pre_layers(const)
        elif self.mode == 'sample_gaussian':
            # Sample independent Gaussian vector for each sample in the batch
            rand_input = torch.randn(x.size(0), 128, device=x.device)
            h_pre = self.pre_layers(rand_input)
            
        h = h_in + h_pre
        return self.post_layers(h)

# 3. Training and Evaluation Functions

def train_model(mode, epochs=5):
    print(f"\n--- Training Model: {mode} ---")
    model = GaussianMiddleInputNet(mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, criterion)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    return history

def evaluate_model(model, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    val_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)
    return val_loss, val_acc

# 4. Run Experiment
modes = ['ones', 'fixed_gaussian', 'sample_gaussian']
results = {}

for mode in modes:
    results[mode] = train_model(mode, epochs=20)

# 5. Plot Results
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
for mode in modes:
    plt.plot(results[mode]['val_acc'], label=mode, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
for mode in modes:
    plt.plot(results[mode]['val_loss'], label=mode, marker='s')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('gaussian_results.png')
print("\nSaved plot to gaussian_results.png")

# Print Summary Table
print("\nFinal Results Summary (after 20 epochs):")
print("-" * 50)
print(f"{'Configuration':<20} | {'Val Loss':<10} | {'Val Acc (%)':<12}")
print("-" * 50)
for mode in modes:
    final_loss = results[mode]['val_loss'][-1]
    final_acc = results[mode]['val_acc'][-1]
    print(f"{mode:<20} | {final_loss:<10.4f} | {final_acc:<12.2f}")
print("-" * 50)
