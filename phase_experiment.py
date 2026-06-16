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

class MiddleInputNet(nn.Module):
    def __init__(self, mode='trainable_pre'):
        super(MiddleInputNet, self).__init__()
        self.mode = mode
        
        # Input projection layer
        self.input_proj = nn.Linear(28 * 28, 128)
        
        # Pre-layers (layers before input injection)
        self.pre_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Register a constant buffer for input to pre-layers
        self.register_buffer('constant_input', torch.ones(1, 128))
        
        # Post-layers (layers after input injection)
        self.post_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        # Flatten input
        x = x.view(-1, 28 * 28)
        h_in = self.input_proj(x)
        
        # Run pre-layers with constant input (matching batch size)
        const = self.constant_input.expand(x.size(0), -1)
        h_pre = self.pre_layers(const)
        
        h = h_in + h_pre
        return self.post_layers(h)

# 3. Training and Evaluation Functions

def train_model(model, epochs=10, description=""):
    print(f"\n--- Training Model: {description} ---")
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.002)
    
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

# 4. Phase-based Experiment Execution

# Phase 1: Pre-train a model to learn the memory bank
print(">>> Phase 1: Training initial model to populate the Memory Bank...")
pre_train_model = MiddleInputNet().to(device)
_ = train_model(pre_train_model, epochs=10, description="Initial Pre-layers Training")

# Extract the learned weights of pre_layers
learned_pre_state_dict = pre_train_model.pre_layers.state_dict()
print("Extracted pre-layer weights successfully.")

# Initialize the three configurations for comparison
results = {}

# Configuration A: Frozen Learned Pre-layers
print("\n>>> Phase 2: Evaluating Frozen Learned Memory Bank...")
model_frozen_learned = MiddleInputNet().to(device)
# Load learned weights
model_frozen_learned.pre_layers.load_state_dict(learned_pre_state_dict)
# Freeze pre-layers
for param in model_frozen_learned.pre_layers.parameters():
    param.requires_grad = False
results['frozen_learned_pre'] = train_model(model_frozen_learned, epochs=10, description="Frozen Learned Memory Bank")

# Configuration B: Frozen Random Pre-layers
print("\n>>> Phase 2: Evaluating Frozen Random Memory Bank...")
model_frozen_random = MiddleInputNet().to(device)
# Freeze pre-layers (random weights)
for param in model_frozen_random.pre_layers.parameters():
    param.requires_grad = False
results['frozen_random_pre'] = train_model(model_frozen_random, epochs=10, description="Frozen Random Memory Bank")

# Configuration C: Trainable Pre-layers (Control Baseline)
print("\n>>> Phase 2: Evaluating Fully Trainable Pre-layers...")
model_trainable = MiddleInputNet().to(device)
results['trainable_pre'] = train_model(model_trainable, epochs=10, description="Fully Trainable Pre-layers")

# 5. Plot Results
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
for key in results:
    plt.plot(results[key]['val_acc'], label=key, marker='o')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
for key in results:
    plt.plot(results[key]['val_loss'], label=key, marker='s')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('phase_results.png')
print("\nSaved plot to phase_results.png")

# Print Summary Table
print("\nFinal Results Summary (after 10 epochs):")
print("-" * 50)
print(f"{'Configuration':<20} | {'Val Loss':<10} | {'Val Acc (%)':<12}")
print("-" * 50)
for key in results:
    final_loss = results[key]['val_loss'][-1]
    final_acc = results[key]['val_acc'][-1]
    print(f"{key:<20} | {final_loss:<10.4f} | {final_acc:<12.2f}")
print("-" * 50)
