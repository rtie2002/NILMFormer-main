
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
    """Prompt user for all three paths interactively."""
    print("========================================")
    print("  NILMFormer Interactive Path Selector  ")
    print("========================================")
    
    # 1. Select Model
    results_dir = ROOT / "result"
    if not results_dir.exists(): results_dir = ROOT / "results"
    model_path = interactive_selection(results_dir, "NILMFormer_*.pt", "Select Model (.pt)")
    
    # 2. Select Data Directory (Tensors)
    tensors_root = ROOT / "prepared_data" / "tensors"
    data_dir = interactive_selection(tensors_root, "*%", "Select Data Directory (Tensors)", is_dir=True)
    
    # 3. Select CSV Path
    csv_root = ROOT / "prepared_data"
    csv_path = interactive_selection(csv_root, "*.csv", "Select Real Power CSV")
    
    return model_path, data_dir, csv_path

def load_model(model_path, device):
    """Load the NILMFormer model from the checkpoint."""
    cfg = NILMFormerConfig(c_in=1, c_embedding=8, c_out=1)
    model = NILMFormer(cfg).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # Check common checkpoint keys
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else (
                 ckpt["best_model_state_dict"] if "best_model_state_dict" in ckpt else ckpt)
    
    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def denormalize(arr, max_val):
    """Convert normalized [0,1] back to Watts."""
    return np.clip(arr * max_val, 0, None)

def visualize_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- NILMFormer Prediction Visualizer ---")
    print(f"Device: {device}")

    # --- 1. Get Paths Interactively ---
    model_path, data_dir, csv_path = get_user_paths()
    if not model_path or not data_dir or not csv_path:
        print("Selection cancelled or failed.")
        return

    # Load Model
    print(f"\nLoading model: {model_path.name}...")
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Tensors
    print(f"Loading test tensors from {data_dir.name}...")
    try:
        test_agg   = torch.load(data_dir / "test_agg.pt",   weights_only=False).numpy()
        test_time  = torch.load(data_dir / "test_time.pt",  weights_only=False).numpy()
        test_power = torch.load(data_dir / "test_power.pt", weights_only=False).numpy()
        test_state = torch.load(data_dir / "test_state.pt", weights_only=False).numpy()
        stats      = torch.load(data_dir / "stats.pt",      weights_only=False)
        
        app_max = float(stats["app_max"])
        agg_max = float(stats["agg_max"])
        print(f"  app_max: {app_max} W | agg_max: {agg_max} W")
    except Exception as e:
        print(f"Error loading tensors: {e}")
        return

    # Prepare 4D data [N, Channels, Vars, Length]
    # NilmFormer expects Aggregate, Temperature (optional), and Time Embeddings
    N, _, L = test_agg.shape
    data_4d = np.zeros((N, 2, 10, L))
    data_4d[:, 0, 0:1, :]  = test_agg
    data_4d[:, 0, 2:10, :] = test_time
    data_4d[:, 1, 0, :]    = test_power[:, 0, :]
    data_4d[:, 1, 1, :]    = test_state[:, 0, :]

    dataset = NILMDataset(data_4d, list_exo_variables=["minute", "hour", "dow", "month"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    preds_raw = []
    trues_raw = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_agg, batch_true, _ in loader:
            batch_agg = batch_agg.to(device)
            out = model(batch_agg)
            preds_raw.append(out.squeeze(1).cpu().numpy())
            trues_raw.append(batch_true.squeeze(1).cpu().numpy())

    preds_w = denormalize(np.concatenate(preds_raw), app_max)
    trues_w = denormalize(np.concatenate(trues_raw), app_max)
    aggs_w  = denormalize(test_agg[:, 0, :], agg_max)

    # --- PLOTTING ---
    plt.style.use('dark_background')
    accent_color = '#00d4ff' # Neon Blue
    gt_color = '#00ff41'     # Matrix Green
    pred_color = '#ff00c1'   # Cyber Pink
    
    # 1. Random Windows Snapshot
    num_samples = 6
    indices = np.random.choice(len(preds_w), num_samples, replace=False)
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0b0b1a')
    fig.suptitle(f"NILMFormer Inference Snapshot\nModel: {model_path.parent.name}", 
                 fontsize=20, color=accent_color, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.25)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor('#121226')
        
        ax.fill_between(range(L), aggs_w[idx], color='#444455', alpha=0.2, label='Aggregate')
        ax.plot(trues_w[idx], color=gt_color, linewidth=2, label='Actual power')
        ax.plot(preds_w[idx], color=pred_color, linestyle='--', linewidth=1.5, label='Predicted power')
        
        mae = np.mean(np.abs(preds_w[idx] - trues_w[idx]))
        ax.set_title(f"Window #{idx} | MAE: {mae:.2f} W", color='white', pad=10)
        ax.set_ylabel("Power (Watts)", fontsize=10)
        ax.grid(color='#2a2a40', linestyle='-', alpha=0.5)
        
        if i == 0:
            legend = ax.legend(loc='upper right', frameon=True, fontsize=9)
            legend.get_frame().set_facecolor('#1a1a35')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 2. Global Overview (Concatenating first 50 windows)
    print("Generating global overview...")
    n_global = min(50, len(preds_w))
    p_global = preds_w[:n_global].flatten()
    t_global = trues_w[:n_global].flatten()
    
    plt.figure(figsize=(18, 6), facecolor='#0b0b1a')
    ax_g = plt.gca()
    ax_g.set_facecolor('#121226')
    ax_g.plot(t_global, color=gt_color, alpha=0.7, label='Actual')
    ax_g.plot(p_global, color=pred_color, label='Predicted', linewidth=1)
    
    ax_g.set_title(f"Continuous Sequence View (First {n_global} Windows)", color=accent_color, fontsize=15)
    ax_g.set_xlabel("Time Steps (minutes)", color='white')
    ax_g.set_ylabel("Power (W)", color='white')
    ax_g.legend()
    ax_g.grid(color='#2a2a40', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\nVisualization complete. If running on a GUI-enabled machine, plots should have appeared.")

if __name__ == "__main__":
    visualize_results()
