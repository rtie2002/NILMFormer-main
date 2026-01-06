import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import sys, os

# Ensure we can import from src
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from src.nilmformer.model import NILMFormer
from src.nilmformer.congif import NILMFormerConfig
from src.helpers.trainer import SeqToSeqTrainer
from src.helpers.metrics import NILMmetrics
from src.helpers.dataset import NILMscaler

class DirectTensorDataset(Dataset):
    """Direct tensor dataset - mimics NILMDataset behavior with explicit concatenation"""
    def __init__(self, tensor_dir, split):
        # Load separate components
        self.agg = torch.load(Path(tensor_dir) / f"{split}_agg.pt", weights_only=False)        # (N, 1, L)
        self.time = torch.load(Path(tensor_dir) / f"{split}_time.pt", weights_only=False)      # (N, 8, L)
        self.target_power = torch.load(Path(tensor_dir) / f"{split}_power.pt", weights_only=False)  # (N, 1, L)
        self.target_state = torch.load(Path(tensor_dir) / f"{split}_state.pt", weights_only=False)  # (N, 1, L)
        
        print(f"[{split}] Loaded: Agg {self.agg.shape}, Time {self.time.shape}")

    def __len__(self):
        return len(self.agg)

    def __getitem__(self, idx):
        # CRITICAL: Concatenate aggregate + time features on-the-fly
        # Agg (1, L) + Time (8, L) -> Input (9, L)
        inputs = torch.cat([self.agg[idx], self.time[idx]], dim=0)  
        
        return (
            inputs,                     # (9, 256)
            self.target_power[idx],     # (1, 256)
            self.target_state[idx]      # (1, 256)
        )


def train(appliance_name, data_dir='prepared_data/tensors'):
    # Load configs
    with open('configs/expes.yaml') as f:
        expes_cfg = yaml.safe_load(f)
    with open('configs/models.yaml') as f:
        model_cfg = yaml.safe_load(f)['NILMFormer']
    
    device = torch.device(expes_cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    tensor_dir = Path(data_dir) / appliance_name
    print(f"Loading data from {tensor_dir}")
    
    train_ds = DirectTensorDataset(tensor_dir, 'train')
    valid_ds = DirectTensorDataset(tensor_dir, 'valid')
    test_ds = DirectTensorDataset(tensor_dir, 'test')
    
    # Verify Item Shape
    sample_input, _, _ = train_ds[0]
    print(f"DEBUG: Single Item Input Shape: {sample_input.shape}")
    if sample_input.shape[0] != 9:
        raise ValueError(f"Dataset returning {sample_input.shape[0]} channels! Expected 9.")

    train_loader = DataLoader(train_ds, batch_size=expes_cfg['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Scaler
    stats = torch.load(tensor_dir / 'stats.pt', weights_only=False)
    scaler = NILMscaler(
        power_scaling_type='MaxScaling',
        appliance_scaling_type='SameAsPower' 
    )
    scaler.is_fitted = True
    scaler.power_stat1 = None 
    scaler.power_stat2 = stats['agg_max']  # Single value, not list
    scaler.appliance_stat1 = [0]  # MaxScaling uses 0 shift
    scaler.appliance_stat2 = [stats['app_max']]  # Single-item list, not [[value]]
    scaler.n_appliance = 1
    
    # Model
    config = NILMFormerConfig()
    config.c_in = 1  # Only the aggregate power channel goes through EmbedBlock
    config.c_embedding = 8  # Number of exogenous (time) features
    for k, v in model_cfg['model_kwargs'].items():
        setattr(config, k, v)
    
    model = NILMFormer(config)
    # Move validation loss to CPU/device correctly
    model.to(device)
    
    # Trainer
    trainer = SeqToSeqTrainer(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        learning_rate=model_cfg['model_training_param']['lr'],
        weight_decay=model_cfg['model_training_param']['wd'],
        criterion=nn.MSELoss(),
        f_metrics=NILMmetrics(),
        training_in_model=model_cfg['model_training_param']['training_in_model'],
        patience_es=expes_cfg['p_es'],
        patience_rlr=expes_cfg['p_rlr'],
        n_warmup_epochs=expes_cfg['n_warmup_epochs'],
        verbose=True,
        plotloss=False,
        save_fig=False,
        path_fig=None,
        device=device,
        all_gpu=expes_cfg['all_gpu'],
        save_checkpoint=True,
        path_checkpoint=f'results/{appliance_name}'
    )
    
    print(f"Starting training for {appliance_name}...")
    trainer.train(expes_cfg['epochs'])
    
    # Evaluate - matching run_one_expe.py pattern (Lines 177-227)
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    print("\nRestoring best model weights...")
    trainer.restore_best_weights()
    
    print("\n[1/2] Evaluating on VALIDATION set...")
    trainer.evaluate(
        valid_loader, 
        scaler=scaler, 
        threshold_small_values=0, 
        save_outputs=True, 
        mask="valid_metrics"
    )
    
    print("\n[2/2] Evaluating on TEST set...")
    trainer.evaluate(
        test_loader, 
        scaler=scaler, 
        threshold_small_values=0, 
        save_outputs=True, 
        mask="test_metrics"
    )
    
    # Display results - matching run_one_expe.py output
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Timestamp-level metrics
    if "valid_metrics_timestamp" in trainer.log:
        print("\n📊 Validation Set (Timestamp-level):")
        for k, v in trainer.log["valid_metrics_timestamp"].items():
            print(f"  {k}: {v:.4f}")
    
    if "test_metrics_timestamp" in trainer.log:
        print("\n📊 Test Set (Timestamp-level):")
        for k, v in trainer.log["test_metrics_timestamp"].items():
            print(f"  {k}: {v:.4f}")
    
    # Window-level metrics (trainer automatically computes these)
    if "valid_metrics_win" in trainer.log:
        print("\n📊 Validation Set (Window-level):")
        for k, v in trainer.log["valid_metrics_win"].items():
            print(f"  {k}: {v:.4f}")
    
    if "test_metrics_win" in trainer.log:
        print("\n📊 Test Set (Window-level):")
        for k, v in trainer.log["test_metrics_win"].items():
            print(f"  {k}: {v:.4f}")
    
    trainer.save()
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE! Results saved to: results/{appliance_name}.pt")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    args = parser.parse_args()
    train(args.appliance)
