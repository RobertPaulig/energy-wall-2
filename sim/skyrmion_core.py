import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# --- ФИЗИКА 2.0: Реальные Материалы (TmIG) ---

PHYSICS_CONFIG = {
    'RRAM_HF02': {
        'name': 'Gen 0: Filamentary RRAM (HfO2)',
        'energy_per_op': 150e-15,  # 150 fJ
        'noise_sigma': 0.08,       # RTN Noise
        'color': '#e74c3c'
    },
    'CDW_TAS2': {
        'name': 'Gen 1: Coherent CDW (TaS2)',
        'energy_per_op': 2e-15,    # 2 fJ (Our current proposal)
        'noise_sigma': 0.005,
        'color': '#f1c40f'
    },
    'SKYRMION_TMIG': {
        'name': 'Gen 2: Topological Skyrmion (TmIG)',
        'energy_per_op': 0.05e-15, # 0.05 fJ (50 aJ) - Реалистично для TmIG!
        'noise_sigma': 0.0005,     # Topologically Protected
        'color': '#2ecc71'
    }
}

# --- Physics Aware Layer ---

class PhysicsLinear(nn.Module):
    def __init__(self, in_features, out_features, config_key):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.config = PHYSICS_CONFIG[config_key]
        self.energy_consumed = 0.0
        
    def forward(self, x):
        # 1. Calculate ideal output
        out = self.linear(x)
        
        # 2. Add Hardware Noise (Gaussian for simplicity, representing thermal/RTN)
        if self.training:
            noise = torch.randn_like(out) * self.config['noise_sigma']
            out = out + noise
            
        # 3. Track Energy
        # Energy = Ops * Energy_per_Op
        # Ops for Linear: 2 * in_features * out_features * batch_size (MACs)
        # We'll approximate just MACs: in * out * batch
        batch_size = x.shape[0]
        ops = self.linear.in_features * self.linear.out_features * batch_size
        self.energy_consumed += ops * self.config['energy_per_op']
        
        return out

class EnergyWallNet(nn.Module):
    def __init__(self, config_key):
        super().__init__()
        self.flatten = nn.Flatten()
        # Simple MLP for MNIST: 784 -> 128 -> 10
        self.fc1 = PhysicsLinear(28*28, 128, config_key)
        self.relu = nn.ReLU()
        self.fc2 = PhysicsLinear(128, 10, config_key)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_total_energy(self):
        return self.fc1.energy_consumed + self.fc2.energy_consumed

# --- Training Loop ---

def train_and_evaluate(config_key, train_loader, test_loader, epochs=5):
    print(f"Starting simulation for: {PHYSICS_CONFIG[config_key]['name']}")
    device = torch.device("cpu") # CPU is fine for this scale and easier for energy tracking logic
    model = EnergyWallNet(config_key).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'energy': [],
        'accuracy': []
    }
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                # Evaluate accuracy
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    # Quick check on a subset of test data to speed up plotting
                    for data_t, target_t in test_loader:
                        output_t = model(data_t)
                        _, predicted = torch.max(output_t.data, 1)
                        total += target_t.size(0)
                        correct += (predicted == target_t).sum().item()
                        break # Just one batch for speed during training curve
                
                acc = 100 * correct / total
                total_energy = model.get_total_energy()
                
                history['energy'].append(total_energy)
                history['accuracy'].append(acc)
                model.train()
                
    print(f"Finished {PHYSICS_CONFIG[config_key]['name']}. Final Energy: {history['energy'][-1]:.2e} J")
    return history

def main():
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download to local folder
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading/Loading MNIST...")
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    results = {}
    
    # Run simulations
    for key in PHYSICS_CONFIG.keys():
        results[key] = train_and_evaluate(key, train_loader, test_loader, epochs=1) # 1 epoch is enough for demo
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    for key, data in results.items():
        config = PHYSICS_CONFIG[key]
        # Convert Energy to Joules for plot (already in Joules)
        # But maybe scale to pJ or nJ for readability? 
        # User asked for Log Scale X-Axis, so raw Joules is fine, just need log scale.
        
        plt.plot(data['energy'], data['accuracy'], label=config['name'], color=config['color'], linewidth=2)
        
    plt.xscale('log')
    plt.xlabel('Total Energy Consumed (Joules) [Log Scale]')
    plt.ylabel('Accuracy on MNIST (%)')
    plt.title('Energy Wall Roadmap: The Path to Skyrmion Computing')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    output_file = 'energy_wall_roadmap.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()
