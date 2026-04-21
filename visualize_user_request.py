
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
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

def normalize_name(name):
    """Convert WashingMachine to washing_machine for path matching."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_user_paths():
    """Step-by-step selector: Appliance -> Window -> Model -> Test Set."""
    print("========================================")
    print("  NILMFormer Professional Selector      ")
    print("========================================")
    
    results_dir = ROOT / "result"
    if not results_dir.exists(): results_dir = ROOT / "results"

    # 1. Select Appliance
    # Extracts "Dishwasher", "WashingMachine" etc from UKDALE_Dishwasher_...
    all_folders = [d.name for d in results_dir.iterdir() if d.is_dir() and "_" in d.name]
    appliances = sorted(list(set([f.split("_")[1] for f in all_folders])))
    
    print("\n--- Select Appliance ---")
    for i, app in enumerate(appliances):
        print(f"  [{i+1:>2}] {app}")
    app_idx = int(input(f"Select appliance (1-{len(appliances)}): ") or 1) - 1
    selected_app = appliances[app_idx]

    # 2. Select Window Size
    # Find all window size folders for this appliance
    app_folders = [d for d in results_dir.iterdir() if d.is_dir() and f"_{selected_app}_" in d.name]
    windows = set()
    for f in app_folders:
        for sub in f.iterdir():
            if sub.is_dir() and sub.name.isdigit():
                windows.add(sub.name)
    windows = sorted(list(windows), key=int)
    
    print(f"\n--- Select Window Size for {selected_app} ---")
    for i, win in enumerate(windows):
        print(f"  [{i+1:>2}] {win}")
    win_idx = int(input(f"Select window size (1-{len(windows)}): ") or 1) - 1
    selected_win = windows[win_idx]

    # 3. Select Specific Model (Percentage)
    print(f"\n--- Select Model Version (Filtered) ---")
    # Find all .pt files matching selected_app and selected_win
    matching_models = sorted(list(results_dir.rglob(f"*_{selected_app}_*/{selected_win}/NILMFormer_*.pt")))
    for i, m in enumerate(matching_models):
        # Show path relative to result/ for clarity
        rel = m.relative_to(results_dir)
        print(f"  [{i+1:>2}] {rel}")
    m_idx = int(input(f"Select model (1-{len(matching_models)}): ") or 1) - 1
    model_path = matching_models[m_idx]

    # 4. Select Test Dataset
    # Default to 0% baseline
    norm_app = normalize_name(selected_app)
    default_test = ROOT / "prepared_data" / "tensors" / selected_win / norm_app / "0%"
    
    print(f"\n--- Test Dataset Selection ---")
    try:
        rel_test = default_test.relative_to(ROOT)
    except ValueError:
        rel_test = default_test
    print(f"Default (Baseline 0%): {rel_test if default_test.exists() else 'Not found'}")
    
    custom_input = input("\n[Press Enter] to use Baseline, or [Paste Tensor Folder Path] for custom test set: ").strip()
    if not custom_input:
        data_dir = default_test
    else:
        # User pasted a string, handle absolute or relative
        raw_path = Path(custom_input.replace('"', '')) # Strip quotes
        
        # Smart detection: If user pasted a CSV file instead of folder
        if raw_path.is_file():
            print(f"⚠️ You pasted a file path. Switching to its directory...")
            data_dir = raw_path.parent
        else:
            data_dir = raw_path
            
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir

    # 5. Smart CSV & Tensor Validation
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return None, None, None, None, None
        
    # If the folder doesn't contain test_agg.pt, maybe it's the wrong level?
    if not (data_dir / "test_agg.pt").exists():
        print(f"⚠️ Warning: No tensors found in {data_dir.name}")
        # Try to suggest correct folder based on appliance and window
        suggested = ROOT / "prepared_data" / "tensors" / selected_win / norm_app / "0%"
        print(f"Suggested tensor folder: {suggested}")
    
    # 6. Locate CSV (Automatic - Smart Search)
    csv_candidates = [
        f"{norm_app}_test__realPower.csv",             # washing_machine_...
        f"{norm_app.replace('_', '')}_test__realPower.csv", # washingmachine_...
        f"{selected_app.lower()}_test__realPower.csv"   # washingmachine_...
    ]
    
    csv_path = None
    for cand in csv_candidates:
        p = ROOT / "prepared_data" / cand
        if p.exists():
            csv_path = p
            break
            
    if not csv_path:
        # Final fallback: search for anything containing the appliance name
        csv_search_key = norm_app.replace('_', '')
        csvs = list((ROOT / "prepared_data").glob(f"*{csv_search_key}*.csv"))
        if not csvs:
            csvs = list((ROOT / "prepared_data").glob(f"*{norm_app}*.csv"))
        csv_path = csvs[0] if csvs else None

    print(f"\n✅ Selection Finalized:")
    print(f"   Model: {model_path}")
    print(f"   Data : {data_dir}")
    print(f"   CSV  : {csv_path.name if csv_path else 'None'}")
    
    return model_path, data_dir, csv_path, str(selected_app), str(selected_win)

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

from matplotlib.widgets import Button, Slider

class InteractiveBrowser:
    def __init__(self, preds, trues, aggs, baseline_preds, app_name, model_info):
        self.preds = preds
        self.trues = trues
        self.aggs = aggs
        self.baseline_preds = baseline_preds
        self.app_name = app_name
        self.model_info = model_info
        self.total = len(preds)
        
        # UI State
        on_windows = [i for i, t in enumerate(trues) if np.max(t) > 20] # Filter empty
        self.curr_idx = on_windows[0] if on_windows else 0
        self.auto_scale = True
        self.double_window = False
        
        # Setup Figure
        plt.style.use('bmh') # Academic look
        self.fig = plt.figure(figsize=(10, 10)) # More square for interactive view
        gs = plt.GridSpec(2, 1, height_ratios=[12, 1])
        self.ax = self.fig.add_subplot(gs[0])
        plt.subplots_adjust(bottom=0.25, left=0.07, right=0.95, top=0.9)
        
        # Add Slider for Fast Scrolling
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Window ', 0, self.total-1, valinit=self.curr_idx, valfmt='%0.0f')
        self.slider.on_changed(self.on_slider)

        # Add Controls Buttons
        ax_prev = plt.axes([0.15, 0.03, 0.08, 0.05])
        ax_next = plt.axes([0.24, 0.03, 0.08, 0.05])
        ax_fit  = plt.axes([0.35, 0.03, 0.12, 0.05])
        
        self.btn_prev = Button(ax_prev, '<< Prev', color='white', hovercolor='0.95')
        self.btn_next = Button(ax_next, 'Next >>', color='white', hovercolor='0.95')
        self.btn_fit  = Button(ax_fit, 'Toggle Auto-Fit', color='white', hovercolor='0.95')
        
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)
        self.btn_fit.on_clicked(self.toggle_fit)
        
        ax_double = plt.axes([0.48, 0.03, 0.15, 0.05])
        self.btn_double = Button(ax_double, 'Double Window: OFF', color='white', hovercolor='0.95')
        self.btn_double.on_clicked(self.toggle_double)

        ax_save = plt.axes([0.65, 0.03, 0.15, 0.05])
        self.btn_save = Button(ax_save, '💾 Save for Paper', color='#e1f5fe', hovercolor='0.95')
        self.btn_save.on_clicked(self.save_for_paper)
        
        # Keyboard & Mouse support
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        # Cursor elements (will be re-initialized in update_plot)
        self.cursor_line = None
        self.cursor_text = None
        
        print(f"\n💡 Interaction UI Loaded:")
        print(f"   - [Mouse]: Drag the slider for FAST SCROLLING.")
        print(f"   - [Arrows]: Left/Right to flip windows.")
        print(f"   - [PgUp/PgDn]: Skip 10 windows at a time.")
        print(f"   - [A]: Toggle Auto-scale.")
        print(f"   - [D]: Toggle Double Window mode.")
        print(f"   - [S]: Save high-res square plot for paper.")
        print(f"   - [Hover]: Move cursor over plot to see values.")
        print(f"   - [Hover]: Move cursor over plot to see values.")

        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        idx = int(self.curr_idx)
        
        # Determine data based on mode
        if self.double_window and idx < self.total - 1:
            idx_next = idx + 1
            pred_data = np.concatenate([self.preds[idx], self.preds[idx_next]])
            true_data = np.concatenate([self.trues[idx], self.trues[idx_next]])
            agg_data  = np.concatenate([self.aggs[idx], self.aggs[idx_next]])
            base_data = np.concatenate([self.baseline_preds[idx], self.baseline_preds[idx_next]]) if self.baseline_preds is not None else None
            window_info = f"Windows {idx}-{idx_next}"
        else:
            pred_data = self.preds[idx]
            true_data = self.trues[idx]
            agg_data  = self.aggs[idx]
            base_data = self.baseline_preds[idx] if self.baseline_preds is not None else None
            window_info = f"Window {idx}"
            
        L = len(pred_data)
        self.curr_L = L # Store for hover
        t = range(L)
        
        # Extract Injection percentage from model_info
        inj_match = re.search(r'(\d+%)', str(self.model_info))
        inj_label = inj_match.group(1) if inj_match else "Selected"
        
        # Plotting
        self.ax.fill_between(t, agg_data, color='gray', alpha=0.15, label='Aggregate Power')
        self.ax.plot(t, true_data, color='#1f77b4', linewidth=2.0, label='Ground Truth (Blue)', alpha=0.9)
        self.ax.plot(t, pred_data, color='#d62728', linestyle='--', linewidth=1.5, label=f'Injection Ratio ({inj_label}) (Red)')
        
        if base_data is not None:
            self.ax.plot(t, base_data, color='k', linestyle=':', label='Baseline (0%)', alpha=0.6)
        
        # Stats
        mae = np.mean(np.abs(pred_data - true_data))
        peak_true = np.max(true_data)
        peak_pred = np.max(pred_data)
        
        mode_str = "Double Mode" if self.double_window else "Single Mode"
        # Clean Title & Labels for Paper
        title = f"{self.app_name.upper()}"
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
        self.ax.set_xlabel("Time (minutes)", fontsize=13, fontweight='bold')
        self.ax.set_ylabel("p(W)", fontsize=13, fontweight='bold')
        
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Always start from 0 on X-axis
        self.ax.set_xlim(0, L)
        
        if self.auto_scale:
            ymax = max(np.max(agg_data), np.max(true_data), 100)
            self.ax.set_ylim(-10, ymax * 1.15)
        else:
            self.ax.set_ylim(-20, max(np.max(self.preds), np.max(self.trues)) * 1.3)
            
        # Re-add Cursor elements after clear
        self.cursor_line = self.ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, visible=False)
        self.cursor_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                        verticalalignment='top', fontsize=10, fontweight='bold',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#d62728'),
                                        visible=False)
            
        self.fig.canvas.draw_idle()

    def on_hover(self, event):
        if event.inaxes == self.ax and self.cursor_line is not None:
            x = int(event.xdata + 0.5) if event.xdata is not None else -1
            idx = int(self.curr_idx)
            
            # Use current displayed data for hover values
            if 0 <= x < self.curr_L:
                self.cursor_line.set_xdata([x])
                self.cursor_line.set_visible(True)
                
                # Fetch values from the plot data actually displayed
                # To be efficient we can slice from original data
                if self.double_window and idx < self.total - 1:
                    idx_next = idx + 1
                    full_true = np.concatenate([self.trues[idx], self.trues[idx_next]])
                    full_pred = np.concatenate([self.preds[idx], self.preds[idx_next]])
                else:
                    full_true = self.trues[idx]
                    full_pred = self.preds[idx]

                inj_match = re.search(r'(\d+%)', str(self.model_info))
                inj_label = inj_match.group(1) if inj_match else "Pred"
                
                txt = f"Timestep: {x} min\n"
                txt += f"Ground Truth: {full_true[x]:.1f}W\n"
                txt += f"Injection Ratio ({inj_label}): {full_pred[x]:.1f}W"
                
                self.cursor_text.set_text(txt)
                self.cursor_text.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                self.cursor_line.set_visible(False)
                self.cursor_text.set_visible(False)
                self.fig.canvas.draw_idle()

    def on_slider(self, val):
        self.curr_idx = int(val)
        self.update_plot()

    def next(self, event):
        self.curr_idx = min(self.total - 1, self.curr_idx + 1)
        self.slider.set_val(self.curr_idx)

    def prev(self, event):
        self.curr_idx = max(0, self.curr_idx - 1)
        self.slider.set_val(self.curr_idx)
        
    def toggle_fit(self, event):
        self.auto_scale = not self.auto_scale
        self.update_plot()

    def on_key(self, event):
        if event.key == 'right': self.next(None)
        elif event.key == 'left': self.prev(None)
        elif event.key == 'pageup':
            self.curr_idx = min(self.total - 1, self.curr_idx + 10)
            self.slider.set_val(self.curr_idx)
        elif event.key == 'pagedown':
            self.curr_idx = max(0, self.curr_idx - 10)
            self.slider.set_val(self.curr_idx)
        elif event.key == 'a': self.toggle_fit(None)
        elif event.key == 'd': self.toggle_double(None)
        elif event.key == 's': self.save_for_paper(None)
        
    def toggle_double(self, event):
        self.double_window = not self.double_window
        label = f"Double Window: {'ON' if self.double_window else 'OFF'}"
        self.btn_double.label.set_text(label)
        self.update_plot()
        
    def save_for_paper(self, event):
        """Export a high-quality square plot for paper publication."""
        idx = int(self.curr_idx)
        
        # Prepare data (same as update_plot)
        if self.double_window and idx < self.total - 1:
            idx_next = idx + 1
            pred_data = np.concatenate([self.preds[idx], self.preds[idx_next]])
            true_data = np.concatenate([self.trues[idx], self.trues[idx_next]])
            agg_data  = np.concatenate([self.aggs[idx], self.aggs[idx_next]])
            base_data = np.concatenate([self.baseline_preds[idx], self.baseline_preds[idx_next]]) if self.baseline_preds is not None else None
            tag = f"win_{idx}_{idx_next}"
        else:
            pred_data = self.preds[idx]
            true_data = self.trues[idx]
            agg_data  = self.aggs[idx]
            base_data = self.baseline_preds[idx] if self.baseline_preds is not None else None
            tag = f"win_{idx}"

        inj_match = re.search(r'(\d+%)', str(self.model_info))
        inj_label = inj_match.group(1) if inj_match else "Selected"
        
        # --- Publication-Quality Figure ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Palatino", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        })
        
        export_fig, ext_ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        L = len(pred_data)
        t = range(L)
        
        # --- Color Palette (Curated for contrast & print clarity) ---
        AGG_COLOR      = '#B0BEC5'   # Blue-Grey (visible but subordinate)
        GT_COLOR       = '#1565C0'   # Deep Blue (anchor)
        PROPOSED_COLOR = '#C62828'   # Deep Red (hero)
        BASELINE_COLOR = '#546E7A'   # Slate Gray (clear but secondary)
        
        # Layer 1: Aggregate Power — VISIBLE fill + outline
        ext_ax.fill_between(t, agg_data, color=AGG_COLOR, alpha=0.25, zorder=1)
        ext_ax.plot(t, agg_data, color=AGG_COLOR, linewidth=1.0, alpha=0.6,
                    label='Aggregate Power', zorder=1)
        
        # Layer 2: Ground Truth — solid, thick anchor line
        ext_ax.plot(t, true_data, color=GT_COLOR, linewidth=2.5,
                    label='Ground Truth', zorder=2)
        
        # Layer 3: Proposed Method — dashed, on top
        ext_ax.plot(t, pred_data, color=PROPOSED_COLOR, linestyle='--', linewidth=2.0,
                    label=f'Injection Ratio ({inj_label})', zorder=4)
        
        # Layer 4: Baseline — clearly visible dashed (NOT dotted)
        if base_data is not None:
            ext_ax.plot(t, base_data, color=BASELINE_COLOR, linestyle=(0, (5, 3)),
                        linewidth=1.8, label='Baseline (0%)', alpha=0.85, zorder=3)
            
        # --- Axes & Labels (LaTeX-style) ---
        ext_ax.set_title(self.app_name.upper(), fontsize=18, fontweight='bold', pad=20)
        ext_ax.set_xlabel("Time (minutes)", fontsize=14, fontweight='bold')
        ext_ax.set_ylabel(r"$P$ (W)", fontsize=14, fontweight='bold', rotation=90)
        
        # CRITICAL: Copy exact scale from interactive view (what you see = what you get)
        ext_ax.set_xlim(self.ax.get_xlim())
        ext_ax.set_ylim(self.ax.get_ylim())
        
        # --- Clean Grid & Spines ---
        ext_ax.grid(True, linestyle=':', alpha=0.35, color='#9E9E9E')
        ext_ax.spines['top'].set_visible(False)
        ext_ax.spines['right'].set_visible(False)
        ext_ax.tick_params(axis='both', labelsize=12)
        
        # --- Legend ---
        ext_ax.legend(loc='upper right', fontsize=11, frameon=True, 
                      framealpha=0.95, edgecolor='#BDBDBD', fancybox=True)
        
        export_fig.tight_layout()
        
        # Save
        filename = f"export_{self.app_name.lower()}_{tag}_{inj_label.replace('%', 'pct')}.png"
        export_fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(export_fig)
        
        # Reset rcParams so interactive plot isn't affected
        plt.rcParams.update(plt.rcParamsDefault)
        
        print(f"\n✨ Publication-quality plot saved: {filename}")
        print(f"   DPI: 300 | Size: 8×8 inches | Font: Serif")

def visualize_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- NILMFormer Academic Visualizer ---")

    # --- 1. Get Paths Interactively ---
    model_path, data_dir, csv_path, selected_app, selected_win = get_user_paths()
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
        
        app_max_stat = float(stats["app_max"])
        agg_max_stat = float(stats["agg_max"])
        
        # SMART SCALE DETECTION
        # Check if test_power was scaled by app_max or agg_max
        raw_max = test_power.max()
        if raw_max > 0:
            val_if_app = raw_max * app_max_stat
            val_if_agg = raw_max * agg_max_stat
            
            # If app_max_stat is same as agg_max_stat, no ambiguity
            if abs(app_max_stat - agg_max_stat) < 1.0:
                print("📝 Detection: Appliance and Aggregate share same scale.")
                final_app_max = agg_max_stat
            elif raw_max < 0.3 and val_if_app < 500 and val_if_agg > 1000:
                # SameAsPower mode: raw is small, app_max gives tiny results, agg_max gives realistic Watts
                print(f"📝 Detection: Appliance scaled using Aggregate Max ({agg_max_stat}W)")
                final_app_max = agg_max_stat
            else:
                print(f"📝 Detection: Appliance scaled using individual Max ({app_max_stat}W)")
                final_app_max = app_max_stat
        else:
            final_app_max = app_max_stat

        app_max = final_app_max
        agg_max = agg_max_stat
        
    except Exception as e:
        print(f"Error loading tensors: {e}")
        return

    # Prepare 4D data
    N, _, L = test_agg.shape
    
    # Validation: Ensure windows match model path
    model_win = selected_win
    if str(L) != str(model_win):
        print(f"⚠️  Warning: Tensor window size ({L}) does not match model window size ({model_win}).")
        print(f"Results may be inaccurate or mismatched.")
    else:
        print(f"✅ Window size verified: {L}")

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

    # --- 5. Denormalize Results ---
    # REVERT: Use app_max for ALL appliance predictions.
    # The previous agg_max fix was incorrect because it led to impossible wattage.
    preds_w = denormalize(np.concatenate(preds_raw), app_max) 
    trues_w = denormalize(np.concatenate(trues_raw), app_max) 
    aggs_w  = denormalize(test_agg[:, 0, :], agg_max)

    # --- 6. Load Baseline (0% Model) for Comparison ---
    baseline_preds_w = None
    try:
        baseline_model_pattern = f"*_{selected_app}_*0%/{selected_win}/NILMFormer_0.pt"
        baseline_models = sorted(list((ROOT / "result").rglob(baseline_model_pattern)))
        
        if baseline_models and baseline_models[0].resolve() != model_path.resolve():
            print(f"Loading baseline (0%) model: {baseline_models[0].parent.parent.name}")
            b_model = load_model(baseline_models[0], device)
            b_model.float()
            b_preds_raw = []
            with torch.no_grad():
                for batch_agg, _, _ in loader:
                    batch_agg = batch_agg.to(device).float()
                    out = b_model(batch_agg)
                    b_preds_raw.append(out.squeeze(1).cpu().numpy())
            # Baseline also uses app_max
            baseline_preds_w = denormalize(np.concatenate(b_preds_raw), app_max)
            print("Baseline inference complete.")
        else:
            print("Selected model is already the baseline (0%).")
    except Exception as e:
        print(f"Skipping baseline comparison due to: {e}")

    # Launcher Interactive Browser
    if not model_path or not data_dir:
        print("Selection process incomplete.")
        return
        
    app_name = selected_app
    model_info = f"{model_path.parent.parent.name} ({model_path.parent.name})"
    
    InteractiveBrowser(preds_w, trues_w, aggs_w, baseline_preds_w, app_name, model_info)

if __name__ == "__main__":
    visualize_results()
    print("\nVisualization complete. If running on a GUI-enabled machine, plots should have appeared.")
