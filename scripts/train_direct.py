import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import sys, os
import numpy as np
import pandas as pd

# Ensure we can import from src
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from src.nilmformer.model import NILMFormer
from src.nilmformer.congif import NILMFormerConfig
from src.helpers.trainer import SeqToSeqTrainer
from src.helpers.metrics import NILMmetrics, eval_win_energy_aggregation
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


def load_4d_data_for_eval(tensor_dir):
    """
    Reconstruct 4D numpy array ONLY for eval_win_energy_aggregation()
    This maintains compatibility with run_one_expe.py's evaluation
    """
    test_agg = torch.load(Path(tensor_dir) / "test_agg.pt", weights_only=False).numpy()
    test_power = torch.load(Path(tensor_dir) / "test_power.pt", weights_only=False).numpy()
    test_state = torch.load(Path(tensor_dir) / "test_state.pt", weights_only=False).numpy()
    
    N, _, L = test_agg.shape
    data_4d = np.zeros((N, 2, 2, L), dtype=np.float32)
    
    # [N, 2(agg+app), 2(power+state), L]
    data_4d[:, 0, 0, :] = test_agg[:, 0, :]
    data_4d[:, 1, 0, :] = test_power[:, 0, :]
    data_4d[:, 1, 1, :] = test_state[:, 0, :]
    
    return data_4d


def create_dummy_dates(n_samples, window_size=256):
    """Create dummy dates for eval_win_energy_aggregation"""
    start_date = pd.Timestamp('2020-01-01 00:00:00')
    dates = [start_date + pd.Timedelta(seconds=i * 10 * window_size) for i in range(n_samples)]
    df = pd.DataFrame({'start_date': dates})
    df.index.name = 'ID_PDL'
    return df


def train(appliance_name, data_dir='prepared_data/tensors'):
    # ============================================================
    # STEP 1: Load Configs (same as run_one_expe.py)
    # ============================================================
    with open('configs/expes.yaml') as f:
        expes_cfg = yaml.safe_load(f)
    with open('configs/models.yaml') as f:
        model_cfg = yaml.safe_load(f)['NILMFormer']
    
    device = torch.device(expes_cfg['device'] if torch.cuda.is_available() else 'cpu')
    
    # ============================================================
    # STEP 2: Load Data (using DirectTensorDataset)
    # ============================================================
    tensor_dir = Path(data_dir) / appliance_name
    print(f"Loading data from {tensor_dir}")
    
    train_ds = DirectTensorDataset(tensor_dir, 'train')
    valid_ds = DirectTensorDataset(tensor_dir, 'valid')
    test_ds = DirectTensorDataset(tensor_dir, 'test')
    
    # ============================================================
    # STEP 3: Create DataLoaders (same as expes.py Line 144-150)
    # ============================================================
    train_loader = DataLoader(train_ds, batch_size=expes_cfg['batch_size'], shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # ============================================================
    # STEP 4: Setup Scaler (same as expes.py/run_one_expe.py Line 90-93)
    # ============================================================
    stats = torch.load(tensor_dir / 'stats.pt', weights_only=False)
    scaler = NILMscaler(
        power_scaling_type='MaxScaling',
        appliance_scaling_type='SameAsPower'
    )
    scaler.is_fitted = True
    scaler.power_stat1 = None
    scaler.power_stat2 = [stats['agg_max']]
    scaler.appliance_stat1 = None
    scaler.appliance_stat2 = [stats['app_max']]  # Single list, not [[value]]
    scaler.n_appliance = 1
    
    # ============================================================
    # STEP 5: Create Model (same as get_model_instance in expes.py Line 84-85)
    # ============================================================
    c_in = 1 + 2 * len(expes_cfg.get('list_exo_variables', ['minute', 'hour', 'dow', 'month']))
    inst_model = NILMFormer(
        NILMFormerConfig(
            c_in=1, 
            c_embedding=c_in - 1, 
            **model_cfg['model_kwargs']
        )
    )
    
    # ============================================================
    # STEP 6: Create Trainer (same as expes.py Line 152-172)
    # ============================================================
    model_trainer = SeqToSeqTrainer(
        inst_model,
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
    
    # ============================================================
    # STEP 7: Train Model (same as expes.py Line 174-175)
    # ============================================================
    print(f"Training {appliance_name} | {len(train_ds)} samples | {device}")
    print("Model training...")
    model_trainer.train(expes_cfg['epochs'])
    
    # ============================================================
    # STEP 8: Evaluate Model (same as expes.py Line 177-192)
    # ============================================================
    print("Eval model...")
    model_trainer.restore_best_weights()
    
    # Valid set evaluation (Line 179-185)
    model_trainer.evaluate(
        valid_loader,
        scaler=scaler,
        threshold_small_values=0,
        save_outputs=True,
        mask="valid_metrics",
    )
    
    # Test set evaluation (Line 186-192)
    model_trainer.evaluate(
        test_loader,
        scaler=scaler,
        threshold_small_values=0,
        save_outputs=True,
        mask="test_metrics",
    )
    
    # ============================================================
    # STEP 9: Window Energy Aggregation (same as expes.py Line 210-222)
    # ============================================================
    # Reconstruct 4D data for this function only
    data_test_4d = load_4d_data_for_eval(tensor_dir)
    st_date_test = create_dummy_dates(len(data_test_4d), expes_cfg.get('window_size', 256))
    
    eval_win_energy_aggregation(
        data_test_4d,
        st_date_test,
        model_trainer,
        scaler=scaler,
        metrics=NILMmetrics(round_to=5),
        window_size=expes_cfg.get('window_size', 256),
        freq=expes_cfg.get('sampling_rate', '10s'),
        list_exo_variables=expes_cfg.get('list_exo_variables', ['minute', 'hour', 'dow', 'month']),
        threshold_small_values=0,
    )
    
    # ============================================================
    # STEP 10: Save (same as expes.py Line 224-226)
    # ============================================================
    model_trainer.save()
    print(f"Training and eval completed! Model weights and log save at: results/{appliance_name}.pt")
    
    # ============================================================
    # Display Results
    # ============================================================
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if "valid_metrics_timestamp" in model_trainer.log:
        print("\n📊 Validation (Timestamp):")
        for k, v in model_trainer.log["valid_metrics_timestamp"].items():
            print(f"  {k}: {v:.4f}")
    
    if "test_metrics_timestamp" in model_trainer.log:
        print("\n📊 Test (Timestamp):")
        for k, v in model_trainer.log["test_metrics_timestamp"].items():
            print(f"  {k}: {v:.4f}")
    
    if "valid_metrics_win" in model_trainer.log:
        print("\n📊 Validation (Window):")
        for k, v in model_trainer.log["valid_metrics_win"].items():
            print(f"  {k}: {v:.4f}")
    
    if "test_metrics_win" in model_trainer.log:
        print("\n📊 Test (Window):")
        for k, v in model_trainer.log["test_metrics_win"].items():
            print(f"  {k}: {v:.4f}")
    
    # Aggregated metrics (D/W/ME)
    for freq_agg in ['D', 'W', 'ME']:
        if f"test_metrics_{freq_agg}" in model_trainer.log:
            freq_name = {'D': 'Daily', 'W': 'Weekly', 'ME': 'Monthly'}[freq_agg]
            print(f"\n📊 Test ({freq_name}):")
            for k, v in model_trainer.log[f"test_metrics_{freq_agg}"].items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    args = parser.parse_args()
    train(args.appliance)
