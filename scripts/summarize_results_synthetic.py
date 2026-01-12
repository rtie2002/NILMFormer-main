
import os
import torch
import pandas as pd
from pathlib import Path

def main():
    result_dir = Path("result")
    results = []

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Walk through the result directory
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".pt"):
                file_path = Path(root) / file
                
                # Parse path for metadata: result/UKDALE_{Appliance}_1min_{SyntheticPct}/{window}/{Model}_{Seed}.pt
                parts = file_path.parts
                # Check structure
                if len(parts) >= 4:
                    try:
                        # parts[1] is typically UKDALE_{Appliance}_1min_{SyntheticPct}
                        dataset_app_synth = parts[1]
                        if "_" in dataset_app_synth:
                            # Split: UKDALE_Dishwasher_1min_0%
                            split_parts = dataset_app_synth.split("_")
                            if len(split_parts) >= 4:
                                appliance = split_parts[1]  # Dishwasher
                                synthetic_pct = split_parts[3]  # 0%
                            else:
                                appliance = split_parts[1] if len(split_parts) > 1 else dataset_app_synth
                                synthetic_pct = "?"
                        else:
                            appliance = dataset_app_synth
                            synthetic_pct = "?"
                        
                        # parts[2] is the window size
                        window = parts[2]
                            
                        filename = file.replace(".pt", "")
                        if "_" in filename:
                            model = filename.split("_")[0]
                            seed = filename.split("_")[1]
                        else:
                            model = filename
                            seed = "?"
                            
                        # Load Metrics
                        data = torch.load(file_path, weights_only=False)
                        
                        mae = "N/A"
                        mr = "N/A" # Match Rate / Accuracy
                        
                        if 'test_metrics_timestamp' in data:
                            metrics = data['test_metrics_timestamp']
                            mae = metrics.get('MAE', 'N/A')
                            
                            # Assume MR matches Accuracy or is explicitly MR
                            if 'MR' in metrics:
                                mr = metrics['MR']
                            elif 'ACCURACY' in metrics:
                                mr = metrics['ACCURACY']
                        
                        results.append({
                            "Appliance": appliance,
                            "Synthetic": synthetic_pct,
                            "Window": window,
                            "Model": model,
                            "Seed": seed,
                            "MAE": mae,
                            "MR": mr
                        })
                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # Display Table
    if results:
        df = pd.DataFrame(results)
        
        # Ensure Synthetic is categorical for proper numeric sorting (0, 25, 50, 100)
        # Create a numeric helper to sort "25%" correctly vs "100%"
        def parse_pct(s):
            return int(s.replace('%', '')) if s != '?' else -1
            
        df['SynthNum'] = df['Synthetic'].apply(parse_pct)
        
        # Get unique window sizes sorted numerically
        windows = sorted(df['Window'].unique(), key=lambda x: int(x))
        
        # Pivot the data: Index=[Appliance, Synthetic], Columns=[Window], Values=[MAE, MR]
        # We'll construct the display manually for full control
        
        # Sort rows: Appliance, then Synthetic %
        df = df.sort_values(by=["Appliance", "SynthNum"])
        
        # Get unique keys for rows
        row_keys = df[['Appliance', 'Synthetic', 'SynthNum']].drop_duplicates().values
        
        print("\n" + "="*120)
        # Build Header
        header_top = f"{'Appliance':<15} {'Synthetic':<10}"
        header_bot = f"{'':<15} {'':<10}"
        
        for w in windows:
            header_top += f" | {'Window (' + str(w) + ')':^24}" # Center "Window (128)"
            header_bot += f" | {'MAE':^11} {'MR':^11}"
            
        print(header_top)
        print(header_bot)
        print("="*120)
        
        current_appliance = None
        
        for app, synth, synth_num in row_keys:
            # Print separator between appliances
            if current_appliance != app and current_appliance is not None:
                print("-" * 120)
            current_appliance = app
            
            # Start Row String
            row_str = f"{app:<15} {synth:<10}"
            
            # Fill Columns for each window
            for w in windows:
                # Find the row for this combo
                match = df[(df['Appliance'] == app) & (df['Synthetic'] == synth) & (df['Window'] == w)]
                
                if not match.empty:
                    mae = match.iloc[0]['MAE']
                    mr = match.iloc[0]['MR']
                    
                    mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else str(mae)
                    mr_str = f"{mr:.3f}" if isinstance(mr, (int, float)) else str(mr)
                    
                    row_str += f" | {mae_str:^11} {mr_str:^11}"
                else:
                    row_str += f" | {'-':^11} {'-':^11}"
            
            print(row_str)

        print("="*120 + "\n")
        print("Note: MR displayed as ACCURACY if explicit MR is missing.")
    else:
        print("No result (.pt) files found.")

if __name__ == "__main__":
    main()
