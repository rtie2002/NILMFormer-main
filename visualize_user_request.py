
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# --- Configuration & Paths ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# --- Project imports (ensure these exist in your src folder)
try:
    from src.nilmformer.congif import NILMFormerConfig
    from src.nilmformer.model import NILMFormer
    from src.helpers.dataset import NILMDataset
except ImportError:
    print("Error: Could not import project modules. Ensure you run this script from the project root.")
    sys.exit(1)

def interactive_selection(root_dir, pattern, prompt_text, is_dir=False):
    """Generic helper to let user select a file or directory from a list."""
    if not root_dir.exists():
        print(f"\n⚠️  Directory not found: {root_dir}")
        return None
        
    items = sorted(root_dir.rglob(pattern))
    if is_dir:
        items = [i for i in items if i.is_dir()]
    else:
        items = [i for i in items if i.is_file()]

    if not items:
        print(f"\n❌ No items found matching '{pattern}' in {root_dir}")
        return None

    print(f"\n--- {prompt_text} ---")
    for i, item in enumerate(items):
        try:
            rel_path = item.relative_to(ROOT)
        except ValueError:
            rel_path = item
        print(f"  [{i+1:>2}] {rel_path}")

    while True:
        try:
            choice = input(f"\nSelect number (1-{len(items)}): ").strip()
            if not choice: return items[0]
            choice = int(choice)
            if 1 <= choice <= len(items):
                return items[choice - 1]
        except (ValueError, IndexError):
            print(f"Please enter a number between 1 and {len(items)}.")

def get_user_paths():
    """Prompt user for model and fully auto-select matching data paths."""
    print("========================================")
    print("  NILMFormer One-Click Visualizer       ")
    print("========================================")
    
    # Selection 1: Model
    results_dir = ROOT / "result"
    if not results_dir.exists(): results_dir = ROOT / "results"
    model_path = interactive_selection(results_dir, "NILMFormer_*.pt", "Select Model (.pt)")
    if not model_path: return None, None, None

    # --- AUTO DETECTION ---
    parts = model_path.parts
    win_size = parts[-2]          # "128"
    folder_name = parts[-3]       # "UKDALE_Dishwasher_1min_0%"
    
    tokens = folder_name.split("_")
    appliance = tokens[1].lower()  # "dishwasher"
    percentage = tokens[-1]        # "0%"

    # 1. Detect Tensors
    data_dir = ROOT / "prepared_data" / "tensors" / win_size / appliance / percentage
    if not data_dir.exists():
        # Fallback 1: Try without percentage folder if using original data structure
        data_dir = ROOT / "prepared_data" / "tensors" / win_size / appliance
        if not data_dir.exists():
            print(f"\n⚠️  Could not auto-locate tensors at {data_dir}")
            data_dir = interactive_selection(ROOT / "prepared_data" / "tensors", "*%", "Manually Select Tensors", is_dir=True)
    
    # 2. Detect CSV
    csv_path = ROOT / "prepared_data" / f"{appliance}_test__realPower.csv"
    if not csv_path.exists():
        print(f"\n⚠️  Could not auto-locate CSV at {csv_path}")
        csv_path = interactive_selection(ROOT / "prepared_data", f"*{appliance}*.csv", "Manually Select CSV")

    print(f"\n✅ Auto-selected:")
    print(f"   Model: {model_path.name}")
    print(f"   Data : {data_dir.relative_to(ROOT) if data_dir else 'None'}")
    print(f"   CSV  : {csv_path.name if csv_path else 'None'}")
    
    return model_path, data_dir, csv_path

def load_model(model_path, device):
    """Load the NILMFormer model from the checkpoint."""
    # We need to peek at the model path to get window size if needed, 
    # but NILMFormerConfig usually handles defaults.
    cfg = NILMFormerConfig(c_in=1, c_embedding=8, c_out=1)
    model = NILMFormer(cfg).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else (
                 ckpt["best_model_state_dict"] if "best_model_state_dict" in ckpt else ckpt)
    
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def denormalize(arr, max_val):
    """Convert normalized [0,1] back to Watts."""
    return np.clip(arr * max_val, 0, None)

from matplotlib.widgets import Button

class InteractiveBrowser:
    def __init__(self, preds, trues, aggs, app_name, model_info):
        self.preds = preds
        self.trues = trues
        self.aggs = aggs
        self.app_name = app_name
        self.model_info = model_info
        self.total = len(preds)
        
        # Start at first window where there is actual power (if any)
        on_windows = [i for i, t in enumerate(trues) if np.sum(t) > 5] # >5W threshold
        self.curr_idx = on_windows[0] if on_windows else 0
        
        # Setup Figure (Academic Style)
        plt.style.use('default') # Light theme
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.2)
        
        self.update_plot()
        
        # Add buttons
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, 'Previous', color='#f0f0f0', hovercolor='#e0e0e0')
        self.btn_next = Button(ax_next, 'Next', color='#f0f0f0', hovercolor='#e0e0e0')
        
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)
        
        plt.show()

    def update_plot(self):
        self.ax.clear()
        idx = self.curr_idx
        L = self.preds.shape[1]
        t = range(L)
        
        # Plotting with academic colors
        self.ax.fill_between(t, self.aggs[idx], color='lightgray', alpha=0.3, label='Aggregate Power')
        self.ax.plot(t, self.trues[idx], color='blue', linewidth=1.5, label='Actual Power (GT)')
        self.ax.plot(t, self.preds[idx], color='red', linestyle='--', linewidth=1.2, label='Predicted Power')
        
        mae = np.mean(np.abs(self.preds[idx] - self.trues[idx]))
        self.ax.set_title(f"Window {idx}/{self.total-1} | Appliance: {self.app_name.capitalize()}\nModel: {self.model_info} | MAE: {mae:.2f} W", 
                          fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Time Step (min)", fontsize=10)
        self.ax.set_ylabel("Power (Watts)", fontsize=10)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(loc='upper right', frameon=True)
        self.ax.set_ylim(bottom=-10, top=max(self.aggs[idx].max() * 1.1, 100))
        
        self.fig.canvas.draw_idle()

    def next(self, event):
        self.curr_idx = (self.curr_idx + 1) % self.total
        self.update_plot()

    def prev(self, event):
        self.curr_idx = (self.curr_idx - 1) % self.total
        self.update_plot()

def visualize_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- NILMFormer Academic Visualizer ---")

    # --- 1. Get Paths Interactively ---
    model_path, data_dir, csv_path = get_user_paths()
    if not model_path or not data_dir or not csv_path:
        return

    # --- 2. Build & Load model ---
    try:
        model = load_model(model_path, device)
        model.float() 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 3. Load Tensors ---
    try:
        test_agg   = torch.load(data_dir / "test_agg.pt",   weights_only=False).numpy()
        test_time  = torch.load(data_dir / "test_time.pt",  weights_only=False).numpy()
        test_power = torch.load(data_dir / "test_power.pt", weights_only=False).numpy()
        test_state = torch.load(data_dir / "test_state.pt", weights_only=False).numpy()
        stats      = torch.load(data_dir / "stats.pt",      weights_only=False)
        
        app_max = float(stats["app_max"])
        agg_max = float(stats["agg_max"])
    except Exception as e:
        print(f"Error loading tensors: {e}")
        return

    # Prepare 4D data
    N, _, L = test_agg.shape
    data_4d = np.zeros((N, 2, 10, L))
    data_4d[:, 0, 0:1, :]  = test_agg
    data_4d[:, 0, 2:10, :] = test_time
    data_4d[:, 1, 0, :]    = test_power[:, 0, :]
    data_4d[:, 1, 1, :]    = test_state[:, 0, :]

    dataset = NILMDataset(data_4d, list_exo_variables=["minute", "hour", "dow", "month"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    preds_raw, trues_raw = [], []
    
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        for batch_agg, batch_true, _ in loader:
            batch_agg = batch_agg.to(device).float()
            out = model(batch_agg)
            preds_raw.append(out.squeeze(1).cpu().numpy())
            trues_raw.append(batch_true.squeeze(1).cpu().numpy())

    preds_w = denormalize(np.concatenate(preds_raw), app_max)
    trues_w = denormalize(np.concatenate(trues_raw), app_max)
    aggs_w  = denormalize(test_agg[:, 0, :], agg_max)

    # Launcher Interactive Browser
    app_name = model_path.parts[-3].split("_")[1]
    model_info = f"{model_path.parts[-3]} ({model_path.parts[-2]})"
    
    InteractiveBrowser(preds_w, trues_w, aggs_w, app_name, model_info)

if __name__ == "__main__":
    visualize_results()

    print("\nVisualization complete. If running on a GUI-enabled machine, plots should have appeared.")

if __name__ == "__main__":
    visualize_results()
