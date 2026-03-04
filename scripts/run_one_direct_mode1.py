"""
run_one_direct_mode1.py
=======================
Runs a single NILMFormer experiment using Mode1 tensors.

Mode1 tensor layout:
    prepared_data_Mode1/tensors/{window_size}/{app}/{scenario}/
        train_agg.pt, train_time.pt, train_power.pt, train_state.pt
        test_agg.pt,  test_time.pt,  test_power.pt,  test_state.pt
        stats.pt  → {agg_max, app_max}

Key differences from run_one_direct.py:
  - Uses --scenario instead of --synthetic_pct
  - Tensor base dir is prepared_data_Mode1/tensors/
  - Validation is always 20% carved from the training data of THIS scenario
    (there is no separate "0% real" folder in Mode1)
  - Results saved under result/mode1/{dataset}_{app}_1min_{scenario}/{win}/
"""

import argparse
import yaml
import logging
import numpy as np

from omegaconf import OmegaConf
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    split_train_test_nilmdataset,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


# ── configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def launch_one_experiment(expes_config: OmegaConf):
    import random, os, torch
    import pandas as pd

    random.seed(expes_config.seed)
    np.random.seed(seed=expes_config.seed)
    torch.manual_seed(expes_config.seed)
    os.environ["PYTHONHASHSEED"] = str(expes_config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(expes_config.seed)
        torch.cuda.manual_seed_all(expes_config.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logging.info("Process data ...")

    # ── Tensor directory ─────────────────────────────────────────────────────
    # Use the appliance name from config
    app_folder_name = expes_config.appliance.lower().replace("_", "")
    tensor_dir = Path(
        f"prepared_data_Mode1/tensors"
        f"/{expes_config.window_size}"
        f"/{app_folder_name}"
        f"/{expes_config.scenario}"
    )

    if not tensor_dir.exists():
        raise FileNotFoundError(
            f"Tensor directory not found: {tensor_dir}\n"
            "Run  python scripts/csv_to_tensors_mode1.py  first."
        )

    logging.info(f"Loading tensors from {tensor_dir}")

    # ── Load tensors ─────────────────────────────────────────────────────────
    train_agg   = torch.load(tensor_dir / "train_agg.pt",   weights_only=False).numpy()
    train_time  = torch.load(tensor_dir / "train_time.pt",  weights_only=False).numpy()
    train_power = torch.load(tensor_dir / "train_power.pt", weights_only=False).numpy()
    train_state = torch.load(tensor_dir / "train_state.pt", weights_only=False).numpy()

    test_agg   = torch.load(tensor_dir / "test_agg.pt",   weights_only=False).numpy()
    test_time  = torch.load(tensor_dir / "test_time.pt",  weights_only=False).numpy()
    test_power = torch.load(tensor_dir / "test_power.pt", weights_only=False).numpy()
    test_state = torch.load(tensor_dir / "test_state.pt", weights_only=False).numpy()

    # Load optional validation tensors (if prepared_data validation was used)
    v_agg, v_time, v_power, v_state = (None, None, None, None)
    if (tensor_dir / "valid_agg.pt").exists():
        logging.info("  Loading fixed validation tensors...")
        v_agg   = torch.load(tensor_dir / "valid_agg.pt",   weights_only=False).numpy()
        v_time  = torch.load(tensor_dir / "valid_time.pt",  weights_only=False).numpy()
        v_power = torch.load(tensor_dir / "valid_power.pt", weights_only=False).numpy()
        v_state = torch.load(tensor_dir / "valid_state.pt", weights_only=False).numpy()

    stats   = torch.load(tensor_dir / "stats.pt", weights_only=False)
    agg_max = stats["agg_max"]
    app_max = stats["app_max"]
    logging.info(f"Scaler stats: agg_max={agg_max:.2f}W  app_max={app_max:.2f}W")

    # ── Reconstruct 4D arrays  (N, 2, 10, W) ─────────────────────────────────
    def reconstruct_4d(agg, time_feat, power, state):
        if agg is None: return None
        N, _, L = agg.shape
        data_4d = np.zeros((N, 2, 10, L), dtype=np.float32)
        data_4d[:, 0, 0:1,  :] = agg
        data_4d[:, 0, 2:10, :] = time_feat
        data_4d[:, 1, 0,    :] = power[:, 0, :]
        data_4d[:, 1, 1,    :] = state[:, 0, :]
        return data_4d

    data_train_raw = reconstruct_4d(train_agg, train_time, train_power, train_state)
    data_test      = reconstruct_4d(test_agg,  test_time,  test_power,  test_state)
    data_valid     = reconstruct_4d(v_agg,     v_time,     v_power,     v_state)

    # ── Validation set handling ──────────────────────────────────────────────
    if data_valid is not None:
        # Use the fixed validation set provided
        data_train = data_train_raw
        logging.info(f"Using FIXED validation set: {data_valid.shape}")
    else:
        # Fallback: 20% carved from training data
        logging.info("Validation tensors not found. Falling back to 20% split of training data.")
        dummy_st_date = pd.DataFrame({
            "start_date": pd.date_range("2013-01-01", periods=len(data_train_raw), freq="10s")
        })
        data_train, _, data_valid, _ = split_train_test_nilmdataset(
            data_train_raw,
            dummy_st_date,
            perc_house_test=0.2,
            seed=expes_config.seed,
            shuffle=False,
        )

    logging.info(f"Train : {data_train.shape}")
    logging.info(f"Valid : {data_valid.shape}")
    logging.info(f"Test  : {data_test.shape}")

    # ── st_dates (None — time features come from .pt files) ─────────────────
    st_date_train = None
    st_date_valid = None
    st_date_test  = None
    data          = np.concatenate([data_train, data_valid, data_test], axis=0)
    st_date       = None

    # ── Scaler (pre-fitted with saved max values) ────────────────────────────
    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    scaler.power_stat1     = 0
    scaler.power_stat2     = agg_max
    scaler.appliance_stat1 = [0]
    scaler.appliance_stat2 = [app_max]
    scaler.n_appliance     = 1
    scaler.is_fitted       = True

    # ── Thresholds ────────────────────────────────────────────────────────────
    thresholds = {
        "kettle":          500,
        "washing_machine": 300,
        "dishwasher":      300,
        "microwave":       200,
        "fridge":           50,
    }
    app_key = expes_config.app.lower().replace(" ", "_")
    if app_key == "washingmachine":
        app_key = "washing_machine"
    expes_config.threshold = thresholds.get(app_key, 10)
    expes_config.cutoff    = float(app_max)

    logging.info(f"Threshold: {expes_config.threshold}W  Cutoff: {expes_config.cutoff:.2f}W")

    # ── Pack tuple_data and launch training ──────────────────────────────────
    tuple_data = (
        data_train,
        data_valid,
        data_test,
        data,
        st_date_train,
        st_date_valid,
        st_date_test,
        st_date,
    )

    launch_models_training(tuple_data, scaler, expes_config)

    # ── Display results ───────────────────────────────────────────────────────
    result_file = f"{expes_config.result_path}.pt"
    if Path(result_file).exists():
        logging.info("\n" + "=" * 60)
        logging.info(f"RESULTS: {expes_config.dataset} | {expes_config.appliance} | "
                     f"scenario={expes_config.scenario} | win={expes_config.window_size} | seed={expes_config.seed}")
        logging.info("=" * 60)
        try:
            import torch as _torch
            log = _torch.load(result_file, weights_only=False)
            for section, key in [
                ("test_metrics_timestamp", "Test  (Timestamp)"),
                ("test_metrics_win",       "Test  (Window)   "),
                ("valid_metrics_timestamp","Valid (Timestamp)"),
            ]:
                if section in log:
                    logging.info(f"\n--- {key} ---")
                    for k, v in log[section].items():
                        logging.info(f"  {k}: {v:.6f}")
            for k in ("epoch_best_loss", "value_best_loss", "training_time"):
                if k in log:
                    logging.info(f"  {k}: {log[k]}")
        except Exception as e:
            logging.error(f"Error reading results: {e}")
    else:
        logging.warning(f"Result file not found: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="NILMFormer Mode1 single experiment.")
    parser.add_argument("--dataset",        required=True,  type=str)
    parser.add_argument("--sampling_rate",  required=True,  type=str)
    parser.add_argument("--window_size",    required=True,  type=str)
    parser.add_argument("--appliance",      required=True,  type=str)
    parser.add_argument("--name_model",     required=True,  type=str)
    parser.add_argument("--seed",           required=True,  type=int)
    parser.add_argument("--scenario",       required=True,  type=str,
                        help="Scenario name, e.g. 200k+10k_ordered")
    args = parser.parse_args()

    try:
        window_size = int(args.window_size)
    except ValueError:
        window_size = args.window_size

    # ── Load configs ──────────────────────────────────────────────────────────
    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)
        if args.dataset in datasets_config:
            datasets_config = datasets_config[args.dataset]
        else:
            raise ValueError(f"Dataset {args.dataset} unknown.")

    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)
        if args.name_model in baselines_config:
            expes_config.update(baselines_config[args.name_model])
        else:
            raise ValueError(f"Model {args.name_model} unknown.")

    # ── Normalize appliance name to match config keys (e.g. dishwasher -> Dishwasher) ──
    app_to_key = {
        "dishwasher": "Dishwasher",
        "fridge": "Fridge",
        "kettle": "Kettle",
        "microwave": "Microwave",
        "washingmachine": "WashingMachine",
        "washing_machine": "WashingMachine"
    }
    app_key = app_to_key.get(args.appliance.lower(), args.appliance.capitalize())

    if app_key in datasets_config:
        expes_config.update(datasets_config[app_key])
        # Force the config appliance name back to the model's expected string
        args.appliance = app_key
    else:
        raise ValueError(f"Appliance {args.appliance} (mapped to {app_key}) not in dataset config.")

    # ── Update config ─────────────────────────────────────────────────────────
    expes_config["dataset"]       = args.dataset
    expes_config["appliance"]     = args.appliance
    expes_config["window_size"]   = window_size
    expes_config["sampling_rate"] = args.sampling_rate
    expes_config["seed"]          = args.seed
    expes_config["name_model"]    = args.name_model
    expes_config["scenario"]      = args.scenario

    # ── Result path ───────────────────────────────────────────────────────────
    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}mode1/")
    result_path = create_dir(
        f"{result_path}{args.dataset}_{args.appliance}_1min_{args.scenario}/"
    )
    result_path = create_dir(f"{result_path}{window_size}/")

    expes_config = OmegaConf.create(expes_config)
    expes_config.result_path = f"{result_path}{expes_config.name_model}_{expes_config.seed}"

    logging.info("---- Mode1 Experiment ----")
    logging.info(f"  Dataset      : {args.dataset}")
    logging.info(f"  Appliance    : {args.appliance}")
    logging.info(f"  Window       : {window_size}")
    logging.info(f"  Scenario     : {args.scenario}")
    logging.info(f"  Model        : {args.name_model}")
    logging.info(f"  Seed         : {args.seed}")
    logging.info(f"  Result path  : {expes_config.result_path}")
    logging.info("--------------------------")

    launch_one_experiment(expes_config)


if __name__ == "__main__":
    main()
