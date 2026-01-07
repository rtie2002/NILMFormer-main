
import sys
import numpy as np
import pandas as pd
import torch
import yaml
# removed omegaconf
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    split_train_test_nilmdataset
)

def main():
    # 1. Load Tensor Stats (Direct)
    tensor_dir = Path('prepared_data/tensors/dishwasher')
    print(f"Loading tensors from {tensor_dir}...")
    
    valid_state_tensor = torch.load(tensor_dir / 'valid_state.pt', weights_only=False).numpy()
    
    print(f"Tensor Valid Shape: {valid_state_tensor.shape}")
    print(f"Tensor Valid ON Ratio: {valid_state_tensor.mean():.4%}")
    
    # 2. Load Raw Stats (Expe)
    print("\nLoading Raw Data via UKDALE_DataBuilder...")
    # Config matching Dishwasher
    config = {
        'data_path': 'data', # Assuming standard path
        'app': 'dishwasher',
        'sampling_rate': '1min',
        'window_size': 256,
        'ind_house_train': [1, 3, 4, 5],
        'ind_house_test': [2],
        'seed': 0
    }
    
    data_builder = UKDALE_DataBuilder(
        data_path="data/UKDALE/", # Relative path
        mask_app=config['app'],
        sampling_rate=config['sampling_rate'],
        window_size=config['window_size'],
    )
    
    data_train, st_date_train = data_builder.get_nilm_dataset(
        house_indicies=config['ind_house_train']
    )
    
    # Split
    data_train, st_date_train, data_valid, st_date_valid = (
        split_train_test_nilmdataset(
            data_train,
            st_date_train,
            perc_house_test=0.2,
            seed=config['seed'],
        )
    )
    
    raw_valid_state = data_valid[:, 1, 1, :]
    
    print(f"Raw Valid Shape: {raw_valid_state.shape}")
    print(f"Raw Valid ON Ratio: {raw_valid_state.mean():.4%}")

if __name__ == "__main__":
    main()
