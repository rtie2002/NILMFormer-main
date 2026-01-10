
import os
import torch
import pandas as pd
from pathlib import Path
import re

def main():
    result_dir = Path("result")
    results = []

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY (3 epochs vs 10 epochs comparison)")
    print("="*80)

    # Walk through the result directory
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".pt"):
                file_path = Path(root) / file
                
                # Parse path for metadata: result/UKDALE_{Appliance}_1min/{window}/{Model}_{Seed}_{Epochs}ep.pt
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
                        
                        # Parse filename: NILMFormer_0_3ep or NILMFormer_0_10ep or NILMFormer_0 (old format)
                        # Use regex to extract model, seed, and epochs
                        match = re.match(r"(.+?)_(\d+)(?:_(\d+)ep)?$", filename)
                        if match:
                            model = match.group(1)
                            seed = match.group(2)
                            epochs = match.group(3) if match.group(3) else "unknown"
                        else:
                            model = filename.split("_")[0] if "_" in filename else filename
                            seed = "?"
                            epochs = "unknown"
                            
                        # Load Metrics
                        data = torch.load(file_path, weights_only=False)
                        
                        mae = "N/A"
                        mr = "N/A" # Match Rate / Accuracy
                        best_epoch = "N/A"
                        
                        if 'test_metrics_timestamp' in data:
                            metrics = data['test_metrics_timestamp']
                            mae = metrics.get('MAE', 'N/A')
                            
                            # Assume MR matches Accuracy or is explicitly MR
                            if 'MR' in metrics:
                                mr = metrics['MR']
                            elif 'ACCURACY' in metrics:
                                mr = metrics['ACCURACY']
                        
                        if 'epoch_best_loss' in data:
                            best_epoch = data['epoch_best_loss']
                        
                        results.append({
                            "Appliance": appliance,
                            "Window": window,
                            "Model": model,
                            "Seed": seed,
                            "Epochs": epochs,
                            "Best_Epoch": best_epoch,
                            "MAE": mae,
                            "MR": mr
                        })
                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # Display Table
    if results:
        df = pd.DataFrame(results)
        # Sort for better readability (by Appliance, Window, Epochs)
        df = df.sort_values(by=["Appliance", "Window", "Epochs", "Model", "Seed"])
        
        # Print with separation lines between appliances
        print()
        current_appliance = None
        current_window = None
        
        for idx, row in df.iterrows():
            # Separator between appliances
            if current_appliance != row['Appliance']:
                if current_appliance is not None:
                    print("=" * 100)
                current_appliance = row['Appliance']
                current_window = None
                print(f"\n{'Appliance':<15} {'Window':<10} {'Epochs':<10} {'Best_Ep':<10} {'MAE':<12} {'MR':<12}")
                print("-" * 100)
            
            # Sub-separator between windows within same appliance
            elif current_window != row['Window']:
                current_window = row['Window']
                print("-" * 100)
            
            # Format the values
            mae_str = f"{row['MAE']:.1f}" if isinstance(row['MAE'], (int, float)) else str(row['MAE'])
            mr_str = f"{row['MR']:.3f}" if isinstance(row['MR'], (int, float)) else str(row['MR'])
            best_epoch_str = str(row['Best_Epoch']) if row['Best_Epoch'] != 'N/A' else 'N/A'
            
            print(f"{row['Appliance']:<15} {row['Window']:<10} {row['Epochs']:<10} {best_epoch_str:<10} {mae_str:<12} {mr_str:<12}")
        
        print("\n" + "="*100)
        
        # Summary comparison: 3 epochs vs 10 epochs
        print("\n" + "="*100)
        print("COMPARISON: 3 Epochs vs 10 Epochs")
        print("="*100)
        
        # Filter for 3 and 10 epoch results
        df_3ep = df[df['Epochs'] == '3'].copy()
        df_10ep = df[df['Epochs'] == '10'].copy()
        
        if not df_3ep.empty and not df_10ep.empty:
            print(f"\n{'Appliance':<15} {'Window':<10} {'MAE_3ep':<12} {'MAE_10ep':<12} {'Diff':<12} {'Winner':<10}")
            print("-" * 100)
            
            # Merge on Appliance and Window
            comparison = pd.merge(
                df_3ep[['Appliance', 'Window', 'MAE']],
                df_10ep[['Appliance', 'Window', 'MAE']],
                on=['Appliance', 'Window'],
                suffixes=('_3ep', '_10ep')
            )
            
            for _, row in comparison.iterrows():
                mae_3 = row['MAE_3ep'] if isinstance(row['MAE_3ep'], (int, float)) else float('nan')
                mae_10 = row['MAE_10ep'] if isinstance(row['MAE_10ep'], (int, float)) else float('nan')
                
                if pd.notna(mae_3) and pd.notna(mae_10):
                    diff = mae_10 - mae_3
                    winner = "3ep" if mae_3 < mae_10 else "10ep" if mae_10 < mae_3 else "tie"
                    
                    print(f"{row['Appliance']:<15} {row['Window']:<10} {mae_3:<12.1f} {mae_10:<12.1f} {diff:+12.1f} {winner:<10}")
        
        print("\n" + "="*100)
        print("Note: Positive Diff means 10-epoch MAE is higher (worse)")
        print("      Negative Diff means 10-epoch MAE is lower (better)")
        print("="*100 + "\n")
    else:
        print("No result (.pt) files found.")

if __name__ == "__main__":
    main()
