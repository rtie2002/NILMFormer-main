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

def get_metrics(app, size, scenario_key, window="128"):
    # Reconstruct the folder name used in run_one_direct_mode1.py
    # Format: UKDALE_{app}_1min_{size}_{scenario_key}
    scenario_full = f"{size}_{scenario_key}"
    
    # Map internal app name to folder name (e.g. washing_machine -> WashingMachine)
    app_norm = app.lower().replace("_", "")
    app_map = {
        "washingmachine": "Wash",
        "dishwasher":     "Dish",
        "fridge":         "Frid",
        "microwave":      "Micr",
        "kettle":         "Kett"
    }

    app_folder_map = {
        "washingmachine": "WashingMachine",
        "dishwasher":     "Dishwasher",
        "fridge":         "Fridge",
        "microwave":      "Microwave",
        "kettle":         "Kettle"
    }
    app_display = app_folder_map.get(app_norm, app.capitalize())

    res_dir = Path(f"result/mode1/UKDALE_{app_display}_1min_{scenario_full}/{window}")
    # Handle multiple model seeds if necessary, but keep seed 0 as default
    res_file = res_dir / "NILMFormer_0.pt" 

    if not res_file.exists():
        return None

    try:
        log = torch.load(res_file, weights_only=False)
        metrics = log.get("test_metrics_timestamp", {})
        return {
            "MAE":    metrics.get("MAE"),
            "SAE":    metrics.get("SAE"),
            "RECALL": metrics.get("RECALL"),
            "F1":     metrics.get("F1_SCORE")
        }
    except:
        return "ERR"

def main():
    WINDOWS = ["128", "256", "512"]
    
    for win in WINDOWS:
        print("\n" + "="*135)
        print(f"COMPREHENSIVE RESULTS SUMMARY (Window Size: {win})")
        print("="*135)
        # Header Row 1: Appliance names
        header1 = f"{'Configuration':<35} |"
        # Metrics to show
        metric_keys = ["MAE", "SAE", "REC", "F1"]
        
        for app in APPLIANCES:
            app_short = app.replace("_machine", "").replace("microwave", "micro").replace("dishwasher", "dish")
            header1 += f" {app_short:^25} |"
        print(header1)

        # Header Row 2: Metric names
        header2 = f"{'':<35} |"
        for _ in APPLIANCES:
            for mk in metric_keys:
                header2 += f" {mk:>5}"
            header2 += " |"
        print(header2)
        print("-" * 135)

        found_any_in_win = False
        for size in SIZES:
            for skey, sname in SCENARIO_MAP.items():
                # Special case for Baseline
                display_config = f"{size} | {sname}"
                if size == "200k+0k" and skey == "ordered":
                    display_config = f"{size} | Baseline"
                elif size == "200k+0k" and skey != "ordered":
                    continue 

                line = f"{display_config:<35} |"
                found_data = False
                for app in APPLIANCES:
                    m = get_metrics(app, size, skey, window=win)
                    if m is None:
                        line += f" {'-':^25} |"
                    elif m == "ERR":
                        line += f" {'FAIL':^25} |"
                    else:
                        # Safe formatting for potential None values
                        def f(v, fmt):
                            return f"{v:{fmt}}" if v is not None else "  -  "
                        
                        m_str = f"{f(m['MAE'], '>5.2f')}{f(m['SAE'], '>6.2f')}{f(m['RECALL'], '>7.2f')}{f(m['F1'], '>7.2f')}"
                        line += f" {m_str} |"
                        found_data = True
                        found_any_in_win = True
                
                if found_data or size == "200k+0k":
                    print(line)

        if not found_any_in_win:
            print(f"{' (No results yet for this window size)':^135}")
        print("="*135 + "\n")

if __name__ == "__main__":
    main()
