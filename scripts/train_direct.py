import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import sys, os

sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from src.nilmformer.model import NILMFormer
from src.nilmformer.congif import NILMFormerConfig
from src.helpers.trainer import SeqToSeqTrainer
from src.helpers.metrics import NILMmetrics

class DirectTensorDataset(Dataset):
    """Direct tensor dataset - mimics NILMDataset behavior"""
    def __init__(self, tensor_dir, split):
        # Load separate components like NILMDataset structure
        self.agg = torch.load(Path(tensor_dir) / f"{split}_agg.pt", weights_only=False)        # (N, 1, L)
        self.time = torch.load(Path(tensor_dir) / f"{split}_time.pt", weights_only=False)      # (N, 8, L)
        self.target_power = torch.load(Path(tensor_dir) / f"{split}_power.pt", weights_only=False)  # (N, 1, L)
        self.target_state = torch.load(Path(tensor_dir) / f"{split}_state.pt", weights_only=False)  # (N, 1, L)

    def __len__(self):
        return len(self.agg)

    def __getitem__(self, idx):
        # Concatenate aggregate + time features on-the-fly (like NILMDataset)
        # This ensures exact same behavior as NILMDataset
        inputs = torch.cat([self.agg[idx], self.time[idx]], dim=0)  # (1, L) + (8, L) = (9, L)
        
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
    
    # Data
    tensor_dir = Path(data_dir) / appliance_name
    train_ds = DirectTensorDataset(tensor_dir, 'train')
    valid_ds = DirectTensorDataset(tensor_dir, 'valid')
    test_ds = DirectTensorDataset(tensor_dir, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=expes_cfg['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Scaler - EXACT same as run_one_expe.py (expes.yaml Line 12-13)
    from src.helpers.dataset import NILMscaler
    stats = torch.load(tensor_dir / 'stats.pt', weights_only=False)
    
    scaler = NILMscaler(
        power_scaling_type='MaxScaling',  # ✓ From expes.yaml
        appliance_scaling_type='SameAsPower'  # ✓ From expes.yaml
    )
    scaler.is_fitted = True
    scaler.power_stat1 = None  # Not used in MaxScaling
    scaler.power_stat2 = [stats['agg_max']]  # Max value
    scaler.appliance_stat1 = None  # Not used
    scaler.appliance_stat2 = [[stats['app_max']]]  # Max value
    scaler.n_appliance = 1
    
    # Model - EXACT same configuration as run_one_expe.py
    config = NILMFormerConfig()
    config.c_in = 9  # 1 agg + 8 time features
    config.c_embedding = 8
    for k, v in model_cfg['model_kwargs'].items():
        setattr(config, k, v)
    
    model = NILMFormer(config)
    
    # Trainer - EXACT same as run_one_expe.py (Line 152-172)
    trainer = SeqToSeqTrainer(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        learning_rate=model_cfg['model_training_param']['lr'],
        weight_decay=model_cfg['model_training_param']['wd'],
        criterion=nn.MSELoss(),
        f_metrics=NILMmetrics(),
        training_in_model=model_cfg['model_training_param']['training_in_model'],
        patience_es=expes_cfg['p_es'],  # Early stopping patience
        patience_rlr=expes_cfg['p_rlr'],  # LR scheduler patience
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
    
    print(f"Training {appliance_name} | {len(train_ds)} samples | {device}")
    
    # Train - EXACT same as run_one_expe.py (Line 175)
    trainer.train(expes_cfg['epochs'])
    
    # Evaluate - EXACT same as run_one_expe.py (Line 177-192)
    print("Evaluating model...")
    trainer.restore_best_weights()
    
    # Valid set evaluation
    trainer.evaluate(
        valid_loader,
        scaler=scaler,
        threshold_small_values=0,  # No threshold for now
        save_outputs=True,
        mask="valid_metrics"
    )
    
    # Test set evaluation
    trainer.evaluate(
        test_loader,
        scaler=scaler,
        threshold_small_values=0,
        save_outputs=True,
        mask="test_metrics"
    )
    
    # Save - EXACT same as run_one_expe.py (Line 224)
    trainer.save()
    print(f"✓ Training and evaluation complete! Model saved to results/{appliance_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    args = parser.parse_args()
    train(args.appliance)
