import pandas as pd
import numpy as np
import torch
import os
import argparse
from pathlib import Path

def convert_appliance_data(appliance_name, data_dir='prepared_data', window_size=256, stride=None):
    if stride is None:
        stride = window_size  # Non-overlapping by default
        
    print(f"\n{'='*50}")
    print(f"Processing Appliance: {appliance_name}")
    print(f"{'='*50}")

    # Paths
    base_path = Path(data_dir)
    output_dir = base_path / 'tensors' / appliance_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Files
    files = {
        'train': base_path / f"{appliance_name}_training__realPower.csv",
        'valid': base_path / f"{appliance_name}_validation__realPower.csv",
        'test': base_path / f"{appliance_name}_test__realPower.csv"
    }
    
    # 1. Load Training Data first to calc Stats
    print(f"Loading Training data for stats calculation...")
    if not files['train'].exists():
        print(f"[ERROR] Training file not found: {files['train']}")
        return

    df_train = pd.read_csv(files['train'])
    
    # Identify appliance column (the one that isnt aggregate or time features)
    time_cols = [
        'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    known_cols = set(['aggregate'] + time_cols + ['timestamp', 'time'])
    potential_app_cols = [c for c in df_train.columns if c not in known_cols]
    
    if not potential_app_cols:
         # Fallback: maybe appliance name is not in cols?
         if appliance_name in df_train.columns:
             app_col = appliance_name
         else:
             print(f"[ERROR] Could not identify appliance column. Cols: {df_train.columns}")
             return
    else:
        app_col = potential_app_cols[0]
    
    print(f"[OK] Identified appliance column: '{app_col}'")

    # Calculate Stats (Z-score)
    agg_mean = df_train['aggregate'].mean()
    agg_std = df_train['aggregate'].std()
    app_mean = df_train[app_col].mean()
    app_std = df_train[app_col].std()
    
    print(f"Stats:")
    print(f"  Agg: {agg_mean:.2f} ± {agg_std:.2f}")
    print(f"  App: {app_mean:.2f} ± {app_std:.2f}")
    
    # Save stats for future use (denormalization)
    stats = {
        'agg_mean': agg_mean, 'agg_std': agg_std,
        'app_mean': app_mean, 'app_std': app_std
    }
    torch.save(stats, output_dir / 'stats.pt')

    # 2. Process each split
    for split_name, file_path in files.items():
        print(f"\nProcessing {split_name}...")
        if not file_path.exists():
            print(f"[WARNING] {split_name} file not found. Skipping.")
            continue
            
        if split_name == 'train':
            df = df_train # Already loaded
        else:
            df = pd.read_csv(file_path)
            
        # Normalize
        # Note: Always use training stats!
        df['aggregate'] = (df['aggregate'] - agg_mean) / agg_std
        df[app_col] = (df[app_col] - app_mean) / app_std
        
        # Sliding Window
        n_timesteps = len(df)
        n_windows = (n_timesteps - window_size) // stride + 1
        
        print(f"  Rows: {n_timesteps}, Windows: {n_windows}")
        
        if n_windows <= 0:
            print("  ⚠️ Not enough data for windows.")
            continue

        # Prepare Tensors
        # Input: (N, 9, L) -> Agg + 8 Time
        inputs = np.zeros((n_windows, 9, window_size), dtype=np.float32)
        # Target: (N, 1, L)
        target_power = np.zeros((n_windows, 1, window_size), dtype=np.float32)
        target_state = np.zeros((n_windows, 1, window_size), dtype=np.float32)
        
        # Fill data
        # We can vectorize this for speed? Or just loop for simplicity/memory safety.
        # Let's loop for now to be safe.
        
        # Pre-extract numpy arrays to speed up loop
        agg_vals = df['aggregate'].values
        app_vals = df[app_col].values
        time_vals = df[time_cols].values.T # (8, Total_Len)
        
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            
            # Input Channel 0: Aggregate
            inputs[i, 0, :] = agg_vals[start:end]
            
            # Input Channel 1-8: Time Features
            inputs[i, 1:9, :] = time_vals[:, start:end]
            
            # Targets
            target_power[i, 0, :] = app_vals[start:end]
            
            # State (Threshold > 0.1 sigma? or just > 0 since it is normalized?)
            # Let's use > 0.0 for normalized, or check raw values?
            # Better to calc state from raw values if possible, but here we only have normalized.
            # Assuming if raw > 10W -> ON. 
            # 10W normalized = (10 - mean) / std.
            thresh_raw = 10 # Watts
            thresh_norm = (thresh_raw - app_mean) / app_std
            target_state[i, 0, :] = (target_power[i, 0, :] > thresh_norm).astype(np.float32)

        # Save
        torch.save(torch.from_numpy(inputs), output_dir / f'{split_name}_inputs.pt')
        torch.save(torch.from_numpy(target_power), output_dir / f'{split_name}_power.pt')
        torch.save(torch.from_numpy(target_state), output_dir / f'{split_name}_state.pt')
        
        print(f"  [OK] Saved {split_name} tensors to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True, help='Appliance name (e.g., dishwasher)')
    parser.add_argument('--window_size', type=int, default=256)
    args = parser.parse_args()
    
    convert_appliance_data(args.appliance, window_size=args.window_size)
