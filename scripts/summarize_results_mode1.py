"""
summarize_results_mode1.py
==========================
Scans the result/mode1 folder and generates a MAE Summary Table
formatted exactly like the research paper requirement.
"""

import os
import torch
import numpy as np
from pathlib import Path

# Mapping for scenario internal names to table display names
SCENARIO_MAP = {
    "ordered": "Ordered",
    "shuffled_w600": "Partial S-w600",
    "full_shuffled_w600": "Full S-w600",
    "shuffled_w6000": "Partial S-w6000",
    "full_shuffled_w6000": "Full S-w6000",
    "event_even_v3": "Event Even",
}

APPLIANCES = ["washing_machine", "dishwasher", "fridge", "microwave", "kettle"]
SIZES = ["200k+0k", "200k+10k", "200k+20k", "200k+100k", "200k+200k", "200k+400k"]

def get_mae(app, size, scenario_key, window="128"):
    # Reconstruct the folder name used in run_one_direct_mode1.py
    # Format: UKDALE_{app}_1min_{size}_{scenario_key}
    scenario_full = f"{size}_{scenario_key}"
    
    # Map internal app name to folder name (e.g. washing_machine -> WashingMachine)
    app_norm = app.lower().replace("_", "")
    app_map = {
        "washingmachine": "WashingMachine",
        "dishwasher": "Dishwasher",
        "fridge": "Fridge",
        "microwave": "Microwave",
        "kettle": "Kettle"
    }
    app_display = app_map.get(app_norm, app.capitalize())

    res_dir = Path(f"result/mode1/UKDALE_{app_display}_1min_{scenario_full}/{window}")
    res_file = res_dir / "NILMFormer_0.pt" # Assuming seed 0

    if not res_file.exists():
        return None

    try:
        log = torch.load(res_file, weights_only=False)
        # Use MAE from test_metrics_timestamp
        metrics = log.get("test_metrics_timestamp", {})
        return metrics.get("MAE")
    except:
        return "ERR"

def main():
    WINDOWS = ["128", "256", "512"]
    
    for win in WINDOWS:
        print("\n" + "="*95)
        print(f"MAE SUMMARY TABLE (Window Size: {win})")
        print("="*95)
        header = f"{'Configuration':<35} |"
        app_headers = ["washingmac", "dishwasher", "fridge", "microwave", "kettle"]
        for ah in app_headers:
            header += f" {ah:>10} |"
        print(header)
        print("-" * 95)

        found_any_in_win = False
        for size in SIZES:
            for skey, sname in SCENARIO_MAP.items():
                # Special case for Baseline
                display_config = f"{size} | {sname}"
                if size == "200k+0k" and skey == "ordered":
                    display_config = f"{size} | Baseline"
                elif size == "200k+0k" and skey != "ordered":
                    continue # Only Baseline exists for 0k

                line = f"{display_config:<35} |"
                found_mae = False
                for app in APPLIANCES:
                    mae = get_mae(app, size, skey, window=win)
                    if mae is None:
                        line += f" {'-':>10} |"
                    elif mae == "ERR":
                        line += f" {'FAIL':>10} |"
                    else:
                        line += f" {mae:>10.2f} |"
                        found_mae = True
                        found_any_in_win = True
                
                if found_mae or size == "200k+0k": # Show row if data exists or it's baseline
                    print(line)

        if not found_any_in_win:
            print(f"{' (No results yet for this window size)':^95}")
        print("="*95 + "\n")

if __name__ == "__main__":
    main()
