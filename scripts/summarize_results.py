import torch
import glob
import os

def summarize():
    # Use glob to find all .pt files in result directory
    # Pattern: result/Dataset_Appliance_1min/window/Model_Seed.pt
    search_path = "result/**/*.pt"
    files = glob.glob(search_path, recursive=True)
    
    print("\n" + "="*70)
    print(f"{'Appliance':<20} | {'Seed':<5} | {'MAE (W)':<10} | {'MR':<10}")
    print("-" * 70)
    
    # Sort files to group by appliance
    files.sort()
    
    for file_path in files:
        try:
            # Parse path structure: result\UKDALE_Dishwasher_1min\256\NILMFormer_0.pt
            filename = os.path.basename(file_path) # NILMFormer_0.pt
            name_no_ext = os.path.splitext(filename)[0] # NILMFormer_0
            
            # Extract Seed (last part)
            if "_" in name_no_ext:
                parts = name_no_ext.rsplit('_', 1)
                model = parts[0]
                seed = parts[1]
            else:
                seed = "?"
            
            # Extract Appliance from directory name
            # Directory: UKDALE_Dishwasher_1min
            parent = os.path.dirname(file_path) # 256
            grandparent = os.path.dirname(parent) # result\UKDALE_Dishwasher_1min
            dir_name = os.path.basename(grandparent)
            
            appliance = dir_name
            if dir_name.startswith("UKDALE_") and dir_name.endswith("_1min"):
                appliance = dir_name.replace("UKDALE_", "").replace("_1min", "")
            
            # Load metrics
            log = torch.load(file_path, weights_only=False)
            
            mae = "N/A"
            mr = "N/A"
            
            if 'test_metrics_timestamp' in log:
                metrics = log['test_metrics_timestamp']
                if 'MAE' in metrics:
                    mae = f"{metrics['MAE']:.2f}"
                if 'MR' in metrics:
                    mr = f"{metrics['MR']:.4f}"
            
            # Only print if we found valid metrics
            if mae != "N/A":
                print(f"{appliance:<20} | {seed:<5} | {mae:<10} | {mr:<10}")
                
        except Exception as e:
            # print(f"Error reading {file_path}: {e}")
            pass
            
    print("="*70 + "\n")

if __name__ == "__main__":
    summarize()
