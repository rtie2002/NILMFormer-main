
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
        # Sort for better readability (by Appliance, Synthetic %, then Window size)
        df = df.sort_values(by=["Appliance", "Synthetic", "Window", "Model", "Seed"])
        
        # Print with separation lines between appliances
        print()
        current_appliance = None
        for idx, row in df.iterrows():
            if current_appliance != row['Appliance']:
                if current_appliance is not None:
                    print("-" * 80)
                current_appliance = row['Appliance']
                # Print header for first row or after separator
                if idx == df.index[0] or current_appliance != df.iloc[df.index.get_loc(idx) - 1]['Appliance']:
                    print(f"{'Appliance':<15} {'Synthetic':<10} {'Window':<10} {'Model':<15} {'Seed':<8} {'MAE':<12} {'MR':<12}")
            
            # Format the values
            mae_str = f"{row['MAE']:.1f}" if isinstance(row['MAE'], (int, float)) else str(row['MAE'])
            mr_str = f"{row['MR']:.3f}" if isinstance(row['MR'], (int, float)) else str(row['MR'])
            
            print(f"{row['Appliance']:<15} {row['Synthetic']:<10} {row['Window']:<10} {row['Model']:<15} {row['Seed']:<8} {mae_str:<12} {mr_str:<12}")
        
        print("\n" + "="*80)
        print("Note: MR displayed as ACCURACY if explicit MR is missing.")
        print("="*80 + "\n")
    else:
        print("No result (.pt) files found.")

if __name__ == "__main__":
    main()
