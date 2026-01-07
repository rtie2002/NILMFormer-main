"""
Interactive window-by-window tensor viewer
Use arrow keys or buttons to navigate through windows
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import argparse

class InteractiveTensorViewer:
    def __init__(self, appliance_name, split='train'):
        self.appliance_name = appliance_name
        self.split = split
        self.current_window = 0
        
        # Load data
        tensor_dir = Path('prepared_data/tensors') / appliance_name
        self.agg = torch.load(tensor_dir / f'{split}_agg.pt', weights_only=False).numpy()
        self.power = torch.load(tensor_dir / f'{split}_power.pt', weights_only=False).numpy()
        self.state = torch.load(tensor_dir / f'{split}_state.pt', weights_only=False).numpy()
        self.time_features = torch.load(tensor_dir / f'{split}_time.pt', weights_only=False).numpy()
        self.stats = torch.load(tensor_dir / 'stats.pt', weights_only=False)
        
        self.total_windows = self.agg.shape[0]
        
        # Denormalize
        self.agg_denorm = self.agg * self.stats['agg_max']
        self.power_denorm = self.power * self.stats['app_max']
        
        print(f"Loaded {appliance_name} - {split}")
        print(f"Total windows: {self.total_windows}")
        print(f"Use ← → arrow keys or buttons to navigate")
        print(f"Close window to exit")
        
        # Create figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f'Tensor Viewer - {appliance_name}')
        
        # Add navigation buttons
        ax_prev = plt.axes([0.3, 0.02, 0.15, 0.04])
        ax_next = plt.axes([0.55, 0.02, 0.15, 0.04])
        self.btn_prev = Button(ax_prev, '← Previous')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_prev.on_clicked(self.prev_window)
        self.btn_next.on_clicked(self.next_window)
        
        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Plot first window
        self.update_plot()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()
    
    def update_plot(self):
        idx = self.current_window
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Aggregate Power
        self.axes[0, 0].plot(self.agg_denorm[idx, 0, :], 'b-', linewidth=1)
        self.axes[0, 0].set_title('Aggregate Power', fontweight='bold')
        self.axes[0, 0].set_ylabel('Power (W)')
        self.axes[0, 0].set_xlabel('Timestep (10s intervals)')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].set_ylim(0, self.stats['agg_max'] * 1.1)
        
        # Plot 2: Appliance Power
        self.axes[0, 1].plot(self.power_denorm[idx, 0, :], 'r-', linewidth=1)
        self.axes[0, 1].set_title('Appliance Power', fontweight='bold')
        self.axes[0, 1].set_ylabel('Power (W)')
        self.axes[0, 1].set_xlabel('Timestep (10s intervals)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylim(0, self.stats['app_max'] * 1.1)
        
        # Plot 3: State
        self.axes[1, 0].fill_between(range(256), 0, self.state[idx, 0, :],
                                      where=self.state[idx, 0, :]>0, 
                                      color='green', alpha=0.6, label='ON')
        self.axes[1, 0].set_title('State (ON/OFF)', fontweight='bold')
        self.axes[1, 0].set_ylabel('State')
        self.axes[1, 0].set_xlabel('Timestep (10s intervals)')
        self.axes[1, 0].set_ylim(-0.1, 1.1)
        self.axes[1, 0].set_yticks([0, 1])
        self.axes[1, 0].set_yticklabels(['OFF', 'ON'])
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Time Features
        self.axes[1, 1].plot(self.time_features[idx, 0, :], 'c-', linewidth=0.8, label='minute_sin', alpha=0.8)
        self.axes[1, 1].plot(self.time_features[idx, 2, :], 'm-', linewidth=0.8, label='hour_sin', alpha=0.8)
        self.axes[1, 1].plot(self.time_features[idx, 4, :], 'y-', linewidth=0.8, label='dow_sin', alpha=0.6)
        self.axes[1, 1].set_title('Time Features', fontweight='bold')
        self.axes[1, 1].set_ylabel('Value')
        self.axes[1, 1].set_xlabel('Timestep (10s intervals)')
        self.axes[1, 1].set_ylim(-1.1, 1.1)
        self.axes[1, 1].legend(fontsize=8, loc='upper right')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Update title
        on_ratio = self.state[idx].mean()
        avg_power = self.power_denorm[idx].mean()
        self.fig.suptitle(
            f'Window {idx+1}/{self.total_windows} | '
            f'Avg Appliance Power: {avg_power:.1f}W | '
            f'State ON: {100*on_ratio:.1f}%',
            fontsize=12, fontweight='bold'
        )
        
        self.fig.canvas.draw()
    
    def next_window(self, event=None):
        if self.current_window < self.total_windows - 1:
            self.current_window += 1
            self.update_plot()
    
    def prev_window(self, event=None):
        if self.current_window > 0:
            self.current_window -= 1
            self.update_plot()
    
    def on_key(self, event):
        if event.key == 'right':
            self.next_window()
        elif event.key == 'left':
            self.prev_window()
        elif event.key == 'escape' or event.key == 'q':
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive tensor viewer')
    parser.add_argument('--appliance', type=str, required=True, help='Appliance name')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'valid', 'test'], help='Which split')
    
    args = parser.parse_args()
    
    viewer = InteractiveTensorViewer(args.appliance, args.split)
