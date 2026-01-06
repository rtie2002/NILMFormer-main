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
        'min_on_duration': 180,
        'min_off_duration': 180,
        'min_activation_time': 12
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
        'min_on_duration': 6,
        'min_off_duration': 1,
        'min_activation_time': 1
    },
    'washing_machine': {
        'min_threshold': 20,
        'max_threshold': 2500,
        'min_on_duration': 180,
        'min_off_duration': 16,
        'min_activation_time': 12
    },
    'microwave': {
        'min_threshold': 200,
        'max_threshold': 3000,
        'min_on_duration': 1,
        'min_off_duration': 3,
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


def convert_appliance_data(appliance_name, data_dir='prepared_data', window_size=256, stride=None):
    if stride is None:
        stride = window_size  # Non-overlapping by default
        
    print(f"\n{'='*50}")
    print(f"Processing Appliance: {appliance_name}")
    print(f"{'='*50}")

    # Get appliance parameters
    if appliance_name not in APPLIANCE_PARAMS:
        print(f"[WARNING] No parameters found for '{appliance_name}'. Using dishwasher defaults.")
        params = APPLIANCE_PARAMS['dishwasher']
    else:
        params = APPLIANCE_PARAMS[appliance_name]
    
    print(f"Using NILMFormer parameters:")
    print(f"  Min threshold: {params['min_threshold']} W")
    print(f"  Max threshold: {params['max_threshold']} W")
    print(f"  Min ON: {params['min_on_duration']} samples (10s each)")
    print(f"  Min OFF: {params['min_off_duration']} samples")

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
    # From expes.yaml: power_scaling_type = MaxScaling
    agg_max = df_train['aggregate'].max()
    app_max = df_train[app_col].max()
    
    print(f"Stats (MaxScaling):")
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
        print(f"\nProcessing {split_name}...")
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

        # Prepare Tensors
        inputs = np.zeros((n_windows, 9, window_size), dtype=np.float32)
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
            
            # Input
            inputs[i, 0, :] = agg_vals[start:end]
            inputs[i, 1:9, :] = time_vals[:, start:end]
            
            # Targets
            target_power[i, 0, :] = app_vals[start:end]
            target_state[i, 0, :] = final_status[start:end]  # Use NILMFormer state

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
