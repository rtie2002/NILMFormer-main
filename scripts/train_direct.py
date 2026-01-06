import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.nilmformer.model import NILMFormer
from src.nilmformer.congif import NILMFormerConfig

class DirectTensorDataset(Dataset):
    def __init__(self, tensor_dir, split):
        self.inputs = torch.load(Path(tensor_dir) / f"{split}_inputs.pt")
        self.target_power = torch.load(Path(tensor_dir) / f"{split}_power.pt")
        # self.target_state = torch.load(Path(tensor_dir) / f"{split}_state.pt") # Optional

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return inputs and target power
        # Input shape: (9, 256)
        # Target shape: (1, 256)
        return self.inputs[idx], self.target_power[idx]

def train(appliance_name, data_dir='prepared_data/tensors'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    tensor_dir = Path(data_dir) / appliance_name
    if not tensor_dir.exists():
        print(f"❌ Error: processed tensors not found at {tensor_dir}")
        print("Please run scripts/convert_csv_to_pt.py first.")
        return

    # DataLoaders
    print("Loading datasets...")
    train_dataset = DirectTensorDataset(tensor_dir, 'train')
    valid_dataset = DirectTensorDataset(tensor_dir, 'valid')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    # Model Config
    config = NILMFormerConfig()
    config.c_in = 9  # 1 agg + 8 time features
    config.c_out = 1 # 1 appliance power
    config.d_model = 128
    config.n_encoder_layers = 2 # Simplify for quick test
    config.window_size = 256
    
    # Init Model
    model = NILMFormer(config).to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    epochs = 10
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
                
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                valid_loss += criterion(output, target).item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")
    
    print("\n✅ Training Complete!")
    # create results dir
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), f'results/model_{appliance_name}.pth')
    print(f"Model saved to results/model_{appliance_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    args = parser.parse_args()
    
    train(args.appliance)
