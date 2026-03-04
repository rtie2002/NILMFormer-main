"""
csv_to_tensors_mode1.py - VERBOSE VERSION
========================================
Now with detailed file logging to prove different scenarios produce different tensors.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COL_AGG   = "aggregate"
TIME_COLS = ["minute_sin", "minute_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

APPLIANCE_PARAMS = {
    'dishwasher':      {'min_threshold': 10,  'max_threshold': 2500, 'min_on_duration': 1, 'min_off_duration': 1, 'min_activation_time': 1},
    'kettle':          {'min_threshold': 2000,'max_threshold': 3100, 'min_on_duration': 1, 'min_off_duration': 0, 'min_activation_time': 1},
    'fridge':          {'min_threshold': 50,  'max_threshold': 300,  'min_on_duration': 1, 'min_off_duration': 1, 'min_activation_time': 1},
    'washing_machine': {'min_threshold': 20,  'max_threshold': 2500, 'min_on_duration': 1, 'min_off_duration': 1, 'min_activation_time': 1},
    'microwave':       {'min_threshold': 200, 'max_threshold': 3000, 'min_on_duration': 1, 'min_off_duration': 1, 'min_activation_time': 1}
}
APPLIANCE_PARAMS['washingmachine'] = APPLIANCE_PARAMS['washing_machine']

def compute_status(initial_status, min_on, min_off, min_activation_time):
    tmp_status = np.zeros_like(initial_status)
    status_diff = np.diff(initial_status)
    events_idx = status_diff.nonzero()
    events_idx = np.array(events_idx).squeeze()
    if events_idx.ndim == 0 and events_idx.size == 1: events_idx = np.array([events_idx])
    events_idx += 1
    if initial_status[0]: events_idx = np.insert(events_idx, 0, 0)
    if initial_status[-1]: events_idx = np.insert(events_idx, events_idx.size, initial_status.size)
    events_idx = events_idx.reshape((-1, 2))
    on_events, off_events = events_idx[:, 0].copy(), events_idx[:, 1].copy()
    if len(on_events) > 0:
        off_duration = np.insert(on_events[1:] - off_events[:-1], 0, 1000)
        on_events = on_events[off_duration > min_off]
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        on_duration = off_events - on_events
        on_events, off_events = on_events[on_duration >= min_on], off_events[on_duration >= min_on]
    valid_activations = (off_events - on_events) >= min_activation_time
    for on, off in zip(on_events[valid_activations], off_events[valid_activations]):
        tmp_status[on:off] = 1
    return tmp_status

def csv_to_windows(csv_path: Path, app_col: str, window_size: int, agg_max: float, app_max: float, params: dict):
    df = pd.read_csv(csv_path, dtype=np.float32)
    agg, power, time = df[COL_AGG].values, df[app_col].values, df[TIME_COLS].values
    initial_status = ((power >= params['min_threshold']) & (power <= params['max_threshold'])).astype(np.int32)
    state = compute_status(initial_status, params['min_on_duration'], params['min_off_duration'], params['min_activation_time']).astype(np.float32)
    agg_norm, power_norm = np.clip(agg / agg_max, 0, 1), np.clip(power / app_max, 0, 1)
    n_wins = len(agg) // window_size
    idx = np.arange(n_wins * window_size).reshape(n_wins, window_size)
    return (torch.tensor(agg_norm[idx][:, np.newaxis, :], dtype=torch.float32),
            torch.tensor(time[idx].transpose(0, 2, 1), dtype=torch.float32),
            torch.tensor(power_norm[idx][:, np.newaxis, :], dtype=torch.float32),
            torch.tensor(state[idx][:, np.newaxis, :], dtype=torch.float32),
            len(df), int(np.sum(power > 0)))

def process_appliance(app_folder: Path, window_sizes: list[int], out_base: Path):
    app_id = app_folder.name.replace("_realPower", "").lower().replace("_", "")
    params = APPLIANCE_PARAMS.get(app_id, APPLIANCE_PARAMS['dishwasher'])
    print(f"\n--- Processing Appliance: {app_id} ---")
    
    test_csv = next(app_folder.glob("*_test__realPower.csv"), None)
    if not test_csv: return
    app_col = pd.read_csv(test_csv, nrows=0).columns[1]
    train_csvs = sorted(app_folder.glob("*_training_*_realPower.csv"))
    valid_csv = next(app_folder.glob("*_validation__realPower.csv"), None) or \
                next(Path("prepared_data").glob(f"{app_id}*_validation__realPower.csv"), None)

    # Global Stats Locking
    stats_files = [test_csv] + [f for f in train_csvs if "200k+0k" in f.name]
    if valid_csv: stats_files.append(valid_csv)
    agg_max = max(pd.read_csv(f, usecols=[COL_AGG]).max()[0] for f in stats_files)
    app_max = max(pd.read_csv(f, usecols=[app_col]).max()[0] for f in stats_files)
    stats = {"agg_max": agg_max, "app_max": app_max}
    print(f"  [Scaler] Lock AggMax={agg_max}, AppMax={app_max}")

    # Process Scenarios
    for train_csv in train_csvs:
        m = re.search(rf"(?:.+)_training_(.+)_realPower\.csv", train_csv.name)
        scenario = m.group(1) if m else "unknown"
        print(f"  > Scenario: {scenario}")
        
        for win in window_sizes:
            out_dir = out_base / str(win) / app_id / scenario
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # UNIQUE training data processing
            t_agg, t_time, t_pow, t_st, n_rows, p_hits = csv_to_windows(train_csv, app_col, win, agg_max, app_max, params)
            print(f"    [WIN={win}] Rows={n_rows}, PowerHits={p_hits} -> Saving to {out_dir.name}")
            
            torch.save(t_agg, out_dir / "train_agg.pt")
            torch.save(t_time, out_dir / "train_time.pt")
            torch.save(t_pow, out_dir / "train_power.pt")
            torch.save(t_st, out_dir / "train_state.pt")
            
            # Test & Valid (cached or re-run)
            test_res = csv_to_windows(test_csv, app_col, win, agg_max, app_max, params)
            torch.save(test_res[0], out_dir / "test_agg.pt")
            torch.save(test_res[1], out_dir / "test_time.pt")
            torch.save(test_res[2], out_dir / "test_power.pt")
            torch.save(test_res[3], out_dir / "test_state.pt")
            
            if valid_csv:
                val_res = csv_to_windows(valid_csv, app_col, win, agg_max, app_max, params)
                torch.save(val_res[0], out_dir / "valid_agg.pt")
                torch.save(val_res[1], out_dir / "valid_time.pt")
                torch.save(val_res[2], out_dir / "valid_power.pt")
                torch.save(val_res[3], out_dir / "valid_state.pt")
            
            torch.save(stats, out_dir / "stats.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, nargs="+", default=[256, 128, 512])
    args = parser.parse_args()
    in_dir = Path("prepared_data_Mode1")
    out_dir = in_dir / "tensors"
    # CLEAR old tensors to be 100% sure
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        print(f"Deleted old tensors at {out_dir}")
    
    app_folders = sorted(f for f in in_dir.glob("*_realPower") if f.is_dir())
    for app in app_folders:
        process_appliance(app, args.window_size, out_dir)
    print("\n--- All Tensors Refreshed Successfully ---")

if __name__ == "__main__":
    main()
