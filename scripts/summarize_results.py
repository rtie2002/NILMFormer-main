
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
                
                # Parse path for metadata: result/UKDALE_{Appliance}_1min/{window}/{Model}_{Seed}.pt
                parts = file_path.parts
                # Check structure
                if len(parts) >= 4:
                    try:
                        # parts[1] is typically UKDALE_{Appliance}_1min
                        dataset_app = parts[1]
                        if "_" in dataset_app:
                            appliance = dataset_app.split("_")[1] # Extract appliance name
                        else:
                            appliance = dataset_app
                        
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
        # Sort for better readability (by Appliance, then Window size)
        df = df.sort_values(by=["Appliance", "Window", "Model", "Seed"])
        
        # Print using pandas to string for nice formatting
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print("Note: MR displayed as ACCURACY if explicit MR is missing.")
        print("="*80 + "\n")
    else:
        print("No result (.pt) files found.")

if __name__ == "__main__":
    main()
