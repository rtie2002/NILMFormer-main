"""
Visualize tensor data from prepared_data/tensors
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def visualize_tensors(appliance_name, split='train', num_windows=5):
    """Visualize tensor data"""
    
    tensor_dir = Path('prepared_data/tensors') / appliance_name
    
    # Load tensors
    agg = torch.load(tensor_dir / f'{split}_agg.pt', weights_only=False).numpy()
    power = torch.load(tensor_dir / f'{split}_power.pt', weights_only=False).numpy()
    state = torch.load(tensor_dir / f'{split}_state.pt', weights_only=False).numpy()
    time_features = torch.load(tensor_dir / f'{split}_time.pt', weights_only=False).numpy()
    stats = torch.load(tensor_dir / 'stats.pt', weights_only=False)
    
    print(f"Loaded {appliance_name} - {split} split")
    print(f"  Total windows: {agg.shape[0]}")
    print(f"  Window size: {agg.shape[2]}")
    print(f"  Stats: agg_max={stats['agg_max']:.2f}W, app_max={stats['app_max']:.2f}W")
    
    # Denormalize for visualization
    agg_denorm = agg * stats['agg_max']
    power_denorm = power * stats['app_max']
    
    # Create figure
    fig, axes = plt.subplots(num_windows, 4, figsize=(20, 3*num_windows))
    fig.suptitle(f'{appliance_name.capitalize()} - {split.capitalize()} Split (First {num_windows} Windows)', 
                 fontsize=16, y=0.995)
    
    for i in range(min(num_windows, agg.shape[0])):
        # Plot 1: Aggregate Power
        axes[i, 0].plot(agg_denorm[i, 0, :], 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel(f'Window {i}\nAggregate (W)', fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(0, stats['agg_max'] * 1.1)
        
        # Plot 2: Appliance Power
        axes[i, 1].plot(power_denorm[i, 0, :], 'r-', linewidth=0.8)
        axes[i, 1].set_ylabel('Appliance (W)', fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim(0, stats['app_max'] * 1.1)
        
        # Plot 3: State
        axes[i, 2].fill_between(range(256), 0, state[i, 0, :], 
                                 where=state[i, 0, :]>0, color='green', alpha=0.5, label='ON')
        axes[i, 2].set_ylabel('State', fontsize=9)
        axes[i, 2].set_ylim(-0.1, 1.1)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_yticks([0, 1])
        
        # Plot 4: Time Features (show minute and hour)
        ax4 = axes[i, 3]
        ax4.plot(time_features[i, 0, :], 'c-', linewidth=0.5, label='min_sin', alpha=0.7)
        ax4.plot(time_features[i, 2, :], 'm-', linewidth=0.5, label='hour_sin', alpha=0.7)
        ax4.set_ylabel('Time Features', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-1.1, 1.1)
        if i == 0:
            ax4.legend(fontsize=7, loc='upper right')
    
    # Set x-labels only for bottom row
    for j in range(4):
        axes[-1, j].set_xlabel('Timestep (10s intervals)', fontsize=9)
    
    # Set column titles
    axes[0, 0].set_title('Aggregate Power', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Appliance Power', fontsize=10, fontweight='bold')
    axes[0, 2].set_title('State (ON/OFF)', fontsize=10, fontweight='bold')
    axes[0, 3].set_title('Time Features', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'tensor_visualization_{appliance_name}_{split}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Create distribution plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle(f'{appliance_name.capitalize()} - {split.capitalize()} Split Distributions', fontsize=14)
    
    # Power distribution
    axes2[0, 0].hist(power_denorm.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
    axes2[0, 0].set_xlabel('Appliance Power (W)')
    axes2[0, 0].set_ylabel('Frequency')
    axes2[0, 0].set_title('Appliance Power Distribution')
    axes2[0, 0].grid(True, alpha=0.3)
    
    # State distribution
    state_counts = [np.sum(state == 0), np.sum(state == 1)]
    axes2[0, 1].bar(['OFF', 'ON'], state_counts, color=['gray', 'green'], alpha=0.7, edgecolor='black')
    axes2[0, 1].set_ylabel('Count')
    axes2[0, 1].set_title(f'State Distribution (ON: {100*state.mean():.2f}%)')
    axes2[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Aggregate vs Appliance scatter
    sample_indices = np.random.choice(agg_denorm.size, min(10000, agg_denorm.size), replace=False)
    axes2[1, 0].scatter(agg_denorm.flatten()[sample_indices], 
                       power_denorm.flatten()[sample_indices], 
                       alpha=0.3, s=1, color='blue')
    axes2[1, 0].set_xlabel('Aggregate Power (W)')
    axes2[1, 0].set_ylabel('Appliance Power (W)')
    axes2[1, 0].set_title('Aggregate vs Appliance')
    axes2[1, 0].grid(True, alpha=0.3)
    
    # Non-zero power distribution
    nonzero_power = power_denorm[power_denorm > 0]
    if len(nonzero_power) > 0:
        axes2[1, 1].hist(nonzero_power, bins=50, color='orange', alpha=0.7, edgecolor='black')
        axes2[1, 1].set_xlabel('Appliance Power (W)')
        axes2[1, 1].set_ylabel('Frequency')
        axes2[1, 1].set_title(f'Non-Zero Power Distribution ({len(nonzero_power)} samples)')
        axes2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save distribution figure
    output_path2 = f'tensor_distributions_{appliance_name}_{split}.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved distributions to: {output_path2}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize tensor data')
    parser.add_argument('--appliance', type=str, required=True, help='Appliance name')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'],
                       help='Which split to visualize')
    parser.add_argument('--num_windows', type=int, default=5, help='Number of windows to show')
    
    args = parser.parse_args()
    
    visualize_tensors(args.appliance, args.split, args.num_windows)
