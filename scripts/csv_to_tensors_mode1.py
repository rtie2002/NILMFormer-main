"""
csv_to_tensors_mode1.py
=======================
Converts Mode1 CSV files from prepared_data_Mode1/{appliance}_realPower/
into the .pt tensor format expected by run_one_direct.py.

CSV Column layout (10 cols):
  aggregate, {appliance}, minute_sin, minute_cos, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

Output tensor layout per scenario folder:
  train_agg.pt    shape (N, 1, W)  - normalized aggregate [0,1]
  train_time.pt   shape (N, 8, W)  - sin/cos time features (already in [-1,1])
  train_power.pt  shape (N, 1, W)  - normalized appliance power [0,1]
  train_state.pt  shape (N, 1, W)  - binary activation state (0 or 1)
  test_agg.pt     shape (M, 1, W)
  test_time.pt    shape (M, 8, W)
  test_power.pt   shape (M, 1, W)
  test_state.pt   shape (M, 1, W)
  stats.pt        dict: {agg_max, app_max}

Usage:
  python scripts/csv_to_tensors_mode1.py --window_size 128 256 512
  python scripts/csv_to_tensors_mode1.py --window_size 128 --appliances dishwasher fridge
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Thresholds (Watts) matching run_one_direct.py ──────────────────────────
THRESHOLDS = {
    "kettle":          500,
    "washing_machine": 300,
    "washingmachine":  300,
    "dishwasher":      300,
    "microwave":       200,
    "fridge":           50,
}

# ── Column indices inside the CSV ───────────────────────────────────────────
COL_AGG   = "aggregate"          # column 0
# column 1 = appliance power (name varies)
TIME_COLS = [                    # columns 2-9
    "minute_sin", "minute_cos",
    "hour_sin",   "hour_cos",
    "dow_sin",    "dow_cos",
    "month_sin",  "month_cos",
]


def csv_to_windows(csv_path: Path, app_col: str, window_size: int,
                   agg_max: float, app_max: float, threshold: float):
    """
    Read a CSV, slide non-overlapping windows of `window_size`,
    return normalized tensors.
    """
    df = pd.read_csv(csv_path, dtype=np.float32)

    agg   = df[COL_AGG].values.astype(np.float32)
    power = df[app_col].values.astype(np.float32)
    time  = df[TIME_COLS].values.astype(np.float32)   # shape (T, 8)

    # Binary state from threshold
    state = (power >= threshold).astype(np.float32)

    # Normalize [0, 1]
    agg_norm   = np.clip(agg   / agg_max, 0, 1)
    power_norm = np.clip(power / app_max, 0, 1)

    T = len(agg)
    n_wins = T // window_size
    if n_wins == 0:
        raise ValueError(f"CSV too short ({T} rows) for window_size={window_size}")

    # Slice into non-overlapping windows
    idx = np.arange(n_wins * window_size).reshape(n_wins, window_size)

    agg_w   = agg_norm[idx][:, np.newaxis, :]   # (N, 1, W)
    power_w = power_norm[idx][:, np.newaxis, :]  # (N, 1, W)
    state_w = state[idx][:, np.newaxis, :]        # (N, 1, W)
    time_w  = time[idx].transpose(0, 2, 1)        # (N, 8, W)

    return (
        torch.tensor(agg_w,   dtype=torch.float32),
        torch.tensor(time_w,  dtype=torch.float32),
        torch.tensor(power_w, dtype=torch.float32),
        torch.tensor(state_w, dtype=torch.float32),
    )


def process_appliance(app_folder: Path, window_sizes: list[int], out_base: Path):
    """Process all CSVs for one appliance folder."""
    # Identify appliance name from folder (e.g. dishwasher_realPower → dishwasher)
    app_name = app_folder.name.replace("_realPower", "").lower()
    app_col  = app_name  # column name in CSV matches appliance name

    threshold = THRESHOLDS.get(app_name, 10)
    print(f"\n{'='*60}")
    print(f"Appliance : {app_name}  (threshold={threshold}W)")
    print(f"{'='*60}")

    # ── Locate files ─────────────────────────────────────────────────────────
    test_csv = app_folder / f"{app_name}_test__realPower.csv"
    if not test_csv.exists():
        print(f"  [SKIP] Test CSV not found: {test_csv}")
        return

    train_csvs = sorted(f for f in app_folder.glob(f"{app_name}_training_*_realPower.csv"))
    if not train_csvs:
        print(f"  [SKIP] No training CSVs found in {app_folder}")
        return

    # ── Compute global max from ALL training files + test file ───────────────
    print("  Computing global max values from all CSVs...")
    all_agg_max   = 0.0
    all_app_max   = 0.0

    for csv in [test_csv] + train_csvs:
        df = pd.read_csv(csv, usecols=[COL_AGG, app_col], dtype=np.float32)
        all_agg_max = max(all_agg_max, float(df[COL_AGG].max()))
        all_app_max = max(all_app_max, float(df[app_col].max()))

    print(f"    agg_max = {all_agg_max:.2f} W")
    print(f"    app_max = {all_app_max:.2f} W")

    stats = {"agg_max": all_agg_max, "app_max": all_app_max}

    # ── Pre-build test & validation tensors once per window size ────────────────
    test_tensors_per_win = {}
    
    # Priority: look in the local appliance folder first
    valid_csv = app_folder / f"{app_name}_validation__realPower.csv"
    if not valid_csv.exists():
        # Fallback to main prepared_data folder
        valid_csv = Path("prepared_data") / f"{app_name}_validation__realPower.csv"
    
    for win in window_sizes:
        print(f"\n  [Test]  window={win}  {test_csv.name}")
        t_agg, t_time, t_power, t_state = csv_to_windows(
            test_csv, app_col, win, all_agg_max, all_app_max, threshold
        )
        
        v_agg, v_time, v_power, v_state = (None, None, None, None)
        if valid_csv.exists():
            print(f"  [Valid] window={win}  {valid_csv.name}")
            v_agg, v_time, v_power, v_state = csv_to_windows(
                valid_csv, app_col, win, all_agg_max, all_app_max, threshold
            )
        else:
            print(f"  [SKIP] Validation CSV not found: {valid_csv}")

        test_tensors_per_win[win] = {
            "test": (t_agg, t_time, t_power, t_state),
            "valid": (v_agg, v_time, v_power, v_state)
        }
        print(f"    → test windows: {t_agg.shape[0]}")
        if v_agg is not None:
            print(f"    → valid windows: {v_agg.shape[0]}")

    # ── Process each training scenario ───────────────────────────────────────
    for train_csv in train_csvs:
        # Parse scenario name from filename
        # e.g. dishwasher_training_200k+10k_ordered_realPower.csv
        #                          ^^^^^^^^^^^^^^^^^^^^
        m = re.search(rf"{app_name}_training_(.+)_realPower\.csv", train_csv.name)
        if not m:
            print(f"  [WARN] Cannot parse scenario from: {train_csv.name}, skipping.")
            continue
        scenario = m.group(1)   # e.g. "200k+10k_ordered"

        print(f"\n  [Train] scenario={scenario}  {train_csv.name}")

        for win in window_sizes:
            out_dir = out_base / str(win) / app_name / scenario
            out_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already done
            if (out_dir / "train_agg.pt").exists() and \
               (out_dir / "test_agg.pt").exists()  and \
               (out_dir / "stats.pt").exists():
                print(f"    window={win}  → already exists, skipping.")
                continue

            # Training tensors
            tr_agg, tr_time, tr_power, tr_state = csv_to_windows(
                train_csv, app_col, win, all_agg_max, all_app_max, threshold
            )
            print(f"    window={win}  → train={tr_agg.shape[0]} windows")

            # Save training tensors
            torch.save(tr_agg,   out_dir / "train_agg.pt")
            torch.save(tr_time,  out_dir / "train_time.pt")
            torch.save(tr_power, out_dir / "train_power.pt")
            torch.save(tr_state, out_dir / "train_state.pt")

            # Save test & validation tensors (shared across scenarios)
            dicts = test_tensors_per_win[win]
            
            t_agg, t_time, t_power, t_state = dicts["test"]
            torch.save(t_agg,   out_dir / "test_agg.pt")
            torch.save(t_time,  out_dir / "test_time.pt")
            torch.save(t_power, out_dir / "test_power.pt")
            torch.save(t_state, out_dir / "test_state.pt")

            v_agg, v_time, v_power, v_state = dicts["valid"]
            if v_agg is not None:
                torch.save(v_agg,   out_dir / "valid_agg.pt")
                torch.save(v_time,  out_dir / "valid_time.pt")
                torch.save(v_power, out_dir / "valid_power.pt")
                torch.save(v_state, out_dir / "valid_state.pt")

            # Save stats
            torch.save(stats, out_dir / "stats.pt")

            print(f"    Saved to {out_dir}")

    print(f"\n  Done: {app_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mode1 CSVs to NILMFormer tensor format."
    )
    parser.add_argument(
        "--input_dir", type=str,
        default="prepared_data_Mode1",
        help="Root folder containing {appliance}_realPower subfolders."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="prepared_data_Mode1/tensors",
        help="Root output folder for tensors (default: prepared_data_Mode1/tensors)."
    )
    parser.add_argument(
        "--window_size", type=int, nargs="+",
        default=[128, 256, 512],
        help="Window size(s) to generate (default: 128 256 512)."
    )
    parser.add_argument(
        "--appliances", type=str, nargs="+",
        default=None,
        help="Subset of appliances to process (default: all found in input_dir)."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}")
        sys.exit(1)

    # Find appliance folders
    app_folders = sorted(f for f in input_dir.iterdir()
                         if f.is_dir() and f.name.endswith("_realPower"))

    if args.appliances:
        app_set = set(a.lower() for a in args.appliances)
        app_folders = [f for f in app_folders
                       if f.name.replace("_realPower", "").lower() in app_set]

    if not app_folders:
        print("ERROR: No appliance folders found.")
        sys.exit(1)

    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {output_dir.resolve()}")
    print(f"Windows: {args.window_size}")
    print(f"Apps   : {[f.name for f in app_folders]}")

    for app_folder in app_folders:
        process_appliance(app_folder, args.window_size, output_dir)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
