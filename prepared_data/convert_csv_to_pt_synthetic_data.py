import pandas as pd
import numpy as np
import torch
import os
import argparse
from pathlib import Path

# Appliance parameters from NILMFormer (preprocessing.py Line 402-438)
# Time parameters are in units of 10sec (minimum sampling rate)
APPLIANCE_PARAMS = {
    'dishwasher': {
        'min_threshold': 10,
        'max_threshold': 2500,
        'min_on_duration': 30,  # Adjusted for 1min data (Original 180 @ 10s = 30mins)
        'min_off_duration': 30, # Adjusted for 1min data (Original 180 @ 10s = 30mins)
        'min_activation_time': 2 # Adjusted for 1min data (Original 12 @ 10s = 2mins)
    },
    'kettle': {
        'min_threshold': 2000,
        'max_threshold': 3100,
        'min_on_duration': 1,
        'min_off_duration': 0,
        'min_activation_time': 1
    },
    'fridge': {
        'min_threshold': 50,
        'max_threshold': 300,
        'min_on_duration': 1, # Adjusted (Original 6 @ 10s = 1min)
        'min_off_duration': 1,
        'min_activation_time': 1
    },
    'washing_machine': {
        'min_threshold': 20,
        'max_threshold': 2500,
        'min_on_duration': 30, # Adjusted (Original 180 @ 10s = 30mins)
        'min_off_duration': 3,  # Adjusted (Original 16 @ 10s = 2.6mins -> ~3mins)
        'min_activation_time': 2 # Adjusted (Original 12 @ 10s = 2mins)
    },
    'microwave': {
        'min_threshold': 200,
        'max_threshold': 3000,
        'min_on_duration': 1,
        'min_off_duration': 1, # Adjusted (Original 3 @ 10s = 30s -> 1min)
        'min_activation_time': 1
    }
}

def compute_status(initial_status, min_on, min_off, min_activation_time):
    """
    NILMFormer's original state detection method (preprocessing.py Line 558-597)
    Filters states based on minimum ON/OFF durations and activation time.
    """
    tmp_status = np.zeros_like(initial_status)
    status_diff = np.diff(initial_status)
    events_idx = status_diff.nonzero()

    events_idx = np.array(events_idx).squeeze()
    events_idx += 1

    if initial_status[0]:
        events_idx = np.insert(events_idx, 0, 0)

    if initial_status[-1]:
        events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

    events_idx = events_idx.reshape((-1, 2))
    on_events = events_idx[:, 0].copy()
    off_events = events_idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]

        on_duration = off_events - on_events
        on_events = on_events[on_duration >= min_on]
        off_events = off_events[on_duration >= min_on]
        assert len(on_events) == len(off_events)

    # Filter activations based on minimum continuous points
    activation_durations = off_events - on_events
    valid_activations = activation_durations >= min_activation_time
    on_events = on_events[valid_activations]
    off_events = off_events[valid_activations]

    for on, off in zip(on_events, off_events):
        tmp_status[on:off] = 1

    return tmp_status


def convert_appliance_data(appliance_name, synthetic_pct, data_dir='prepared_data', window_size=256, stride=None):
    """
    Convert CSV data to PyTorch tensors for a specific appliance and synthetic percentage.
    
    Folder structure: prepared_data/tensors/{window_size}/{appliance}/{synthetic_pct}/
    Example: prepared_data/tensors/256/dishwasher/0%/
    """
    if stride is None:
        stride = window_size  # Non-overlapping by default
        
    print(f"\n{'='*60}")
    print(f"Processing: {appliance_name} | Window: {window_size} | Synthetic: {synthetic_pct}")
    print(f"{'='*60}")

    # Get appliance parameters
    if appliance_name not in APPLIANCE_PARAMS:
        print(f"[WARNING] No parameters found for '{appliance_name}'. Using dishwasher defaults.")
        params = APPLIANCE_PARAMS['dishwasher']
    else:
        params = APPLIANCE_PARAMS[appliance_name]
    
    print(f"Using NILMFormer parameters:")
    print(f"  Min threshold: {params['min_threshold']} W")
    print(f"  Max threshold: {params['max_threshold']} W")
    print(f"  Min ON: {params['min_on_duration']} samples")
    print(f"  Min OFF: {params['min_off_duration']} samples")

    # Paths - NEW STRUCTURE: window_size/appliance/synthetic_pct
    base_path = Path(data_dir)
    output_dir = base_path / 'tensors' / str(window_size) / appliance_name / synthetic_pct
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle filename mismatch for washing machine
    filename_appliance = appliance_name
    if appliance_name == 'washing_machine':
        filename_appliance = 'washingmachine'
    
    # Determine training file based on synthetic percentage
    if synthetic_pct == "0%":
        train_file = base_path / f"{filename_appliance}_training__realPower.csv"
    else:
        train_file = base_path / f"{filename_appliance}_training_synthetic_{synthetic_pct}_realPower.csv"
    
    # Files - test is always the same (NO VALIDATION FILE - it's split from training at runtime)
    files = {
        'train': train_file,
        'test': base_path / f"{filename_appliance}_test__realPower.csv"
    }
    
    print(f"\nData files:")
    print(f"  Train: {files['train'].name}")
    print(f"  Test:  {files['test'].name}")
    
    # 1. Load Training Data first to calc Stats
    print(f"\nLoading Training data for stats calculation...")
    if not files['train'].exists():
        print(f"[ERROR] Training file not found: {files['train']}")
        return

    df_train = pd.read_csv(files['train'])
    
    # Identify appliance column
    time_cols = [
        'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    known_cols = set(['aggregate'] + time_cols + ['timestamp', 'time'])
    potential_app_cols = [c for c in df_train.columns if c not in known_cols]
    
    if not potential_app_cols:
         if appliance_name in df_train.columns:
             app_col = appliance_name
         else:
             print(f"[ERROR] Could not identify appliance column. Cols: {df_train.columns}")
             return
    else:
        app_col = potential_app_cols[0]
    
    print(f"[OK] Identified appliance column: '{app_col}'")

    # Calculate Stats (MaxScaling - same as NILMFormer)
    # CRITICAL: We must calculate stats from REAL Data (Real Train + Real Test) ONLY.
    # We should NOT let Synthetic Data outliers distort the scaling factor.
    # This keeps normalization consistent across 0%, 25%, 100% experiments.
    
    print("\n[STEP 1] Loading REAL data stats (to ensure consistent scaling)...")
    
    real_train_file = base_path / f"{filename_appliance}_training__realPower.csv"
    
    stats_files = {
        'real_train': real_train_file,
        'test': files['test']
    }
    
    all_dfs = []
    for split_name, file_path in stats_files.items():
        if file_path.exists():
            df_split = pd.read_csv(file_path)
            all_dfs.append(df_split)
            print(f"  Loaded {split_name}: {len(df_split)} rows")
        else:
            print(f"  [WARNING] {split_name} file not found for stats: {file_path}")
    
    if not all_dfs:
        print("[ERROR] No data files found for stats!")
        return
    
    # Concatenate ALL data
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate global max from REAL data
    agg_max = df_all['aggregate'].max()
    app_max = df_all[app_col].max()
    
    print(f"\n[STEP 2] Global Stats (Locked to REAL Data distribution):")
    print(f"  Agg max: {agg_max:.2f} W")
    print(f"  App max: {app_max:.2f} W")
    
    # Save stats for scaler
    stats = {
        'agg_max': agg_max,
        'app_max': app_max
    }
    torch.save(stats, output_dir / 'stats.pt')

    # 2. Process each split
    for split_name, file_path in files.items():
        print(f"\n[STEP 3] Processing {split_name}...")
        if not file_path.exists():
            print(f"[WARNING] {split_name} file not found. Skipping.")
            continue
            
        if split_name == 'train':
            df = df_train # Already loaded
        else:
            df = pd.read_csv(file_path)
        
        # Keep RAW power for state detection
        raw_power = df[app_col].values
            
        # Normalize (MaxScaling - same as NILMFormer)
        df['aggregate'] = df['aggregate'] / agg_max
        df[app_col] = df[app_col] / app_max
        
        # Sliding Window
        n_timesteps = len(df)
        n_windows = (n_timesteps - window_size) // stride + 1
        
        print(f"  Rows: {n_timesteps}, Windows: {n_windows}")
        
        if n_windows <= 0:
            print("  [WARNING] Not enough data for windows.")
            continue

        # Prepare Tensors - Split like NILMDataset
        agg_data = np.zeros((n_windows, 1, window_size), dtype=np.float32)  # Aggregate only
        time_data = np.zeros((n_windows, 8, window_size), dtype=np.float32)  # Time features
        target_power = np.zeros((n_windows, 1, window_size), dtype=np.float32)
        target_state = np.zeros((n_windows, 1, window_size), dtype=np.float32)
        
        # Pre-extract arrays
        agg_vals = df['aggregate'].values
        app_vals = df[app_col].values
        time_vals = df[time_cols].values.T # (8, Total_Len)
        
        # Compute state for ENTIRE sequence (NILMFormer style)
        print("  Computing states (NILMFormer method)...")
        initial_status = (
            (raw_power >= params['min_threshold']) &
            (raw_power <= params['max_threshold'])
        ).astype(int)
        
        final_status = compute_status(
            initial_status,
            params['min_on_duration'],
            params['min_off_duration'],
            params['min_activation_time']
        )
        
        print(f"    Initial ON ratio: {initial_status.mean():.2%}")
        print(f"    Final ON ratio: {final_status.mean():.2%}")
        
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            
            # Separate aggregate and time (like NILMDataset)
            agg_data[i, 0, :] = agg_vals[start:end]
            time_data[i, :, :] = time_vals[:, start:end]
            
            # Targets
            target_power[i, 0, :] = app_vals[start:end]
            target_state[i, 0, :] = final_status[start:end]

        # Save - separate aggregate and time features
        torch.save(torch.from_numpy(agg_data), output_dir / f'{split_name}_agg.pt')
        torch.save(torch.from_numpy(time_data), output_dir / f'{split_name}_time.pt')
        torch.save(torch.from_numpy(target_power), output_dir / f'{split_name}_power.pt')
        torch.save(torch.from_numpy(target_state), output_dir / f'{split_name}_state.pt')
        
        print(f"  [OK] Saved {split_name} tensors")
    
    print(f"\n[COMPLETE] All tensors saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV to PyTorch tensors for synthetic data experiments')
    parser.add_argument('--appliance', type=str, help='Appliance name (e.g., dishwasher). If not specified, processes all.')
    parser.add_argument('--window_size', type=int, help='Window size (e.g., 128, 256, 512). If not specified, processes all.')
    parser.add_argument('--synthetic_pct', type=str, help='Synthetic percentage (0%, 25%, 50%, 100%, 200%). If not specified, processes all.')
    parser.add_argument('--all', action='store_true', help='Process all combinations (5 appliances × 3 windows × 5 percentages = 75 experiments)')
    args = parser.parse_args()
    
    # Define all possible values
    all_appliances = ['dishwasher', 'fridge', 'kettle', 'microwave', 'washing_machine']
    all_windows = [128, 256, 512]
    all_percentages = ['0%', '25%', '50%', '100%', '200%']
    
    # Determine what to process
    if args.all:
        appliances = all_appliances
        windows = all_windows
        percentages = all_percentages
        print("\n" + "="*60)
        print("PROCESSING ALL COMBINATIONS")
        print(f"Total experiments: {len(appliances)} × {len(windows)} × {len(percentages)} = {len(appliances)*len(windows)*len(percentages)}")
        print("="*60)
    else:
        appliances = [args.appliance] if args.appliance else all_appliances
        windows = [args.window_size] if args.window_size else all_windows
        percentages = [args.synthetic_pct] if args.synthetic_pct else all_percentages
    
    # Process all combinations
    total = len(appliances) * len(windows) * len(percentages)
    current = 0
    
    for appliance in appliances:
        for window in windows:
            for pct in percentages:
                current += 1
                print(f"\n{'#'*60}")
                print(f"Progress: {current}/{total}")
                print(f"{'#'*60}")
                convert_appliance_data(appliance, pct, window_size=window)
    
    print("\n" + "="*60)
    print("ALL CONVERSIONS COMPLETE!")
    print("="*60)

