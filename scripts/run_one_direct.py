#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import yaml
import logging
import numpy as np

from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.helpers.utils import create_dir
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
)
from src.helpers.dataset import NILMscaler
from src.helpers.expes import launch_models_training


def launch_one_experiment(expes_config: OmegaConf):
    np.random.seed(seed=expes_config.seed)

    logging.info("Process data ...")
    if expes_config.dataset == "UKDALE":
        # ============================================================
        # TENSOR LOADING - Replaces data_builder.get_nilm_dataset()
        # ============================================================
        import torch
        from pathlib import Path
        
        tensor_dir = Path(f'prepared_data/tensors/{expes_config.app.lower()}')
        logging.info(f"Loading tensors from {tensor_dir}")
        
        # Load all tensors
        train_agg = torch.load(tensor_dir / 'train_agg.pt', weights_only=False).numpy()
        train_time = torch.load(tensor_dir / 'train_time.pt', weights_only=False).numpy()
        train_power = torch.load(tensor_dir / 'train_power.pt', weights_only=False).numpy()
        train_state = torch.load(tensor_dir / 'train_state.pt', weights_only=False).numpy()
        
        valid_agg = torch.load(tensor_dir / 'valid_agg.pt', weights_only=False).numpy()
        valid_time = torch.load(tensor_dir / 'valid_time.pt', weights_only=False).numpy()
        valid_power = torch.load(tensor_dir / 'valid_power.pt', weights_only=False).numpy()
        valid_state = torch.load(tensor_dir / 'valid_state.pt', weights_only=False).numpy()
        
        test_agg = torch.load(tensor_dir / 'test_agg.pt', weights_only=False).numpy()
        test_time = torch.load(tensor_dir / 'test_time.pt', weights_only=False).numpy()
        test_power = torch.load(tensor_dir / 'test_power.pt', weights_only=False).numpy()
        test_state = torch.load(tensor_dir / 'test_state.pt', weights_only=False).numpy()
        
        # Reconstruct 4D arrays: (N, 2, 10, window_size)
        def reconstruct_4d(agg, time_feat, power, state):
            N, _, L = agg.shape
            data_4d = np.zeros((N, 2, 10, L))
            data_4d[:, 0, 0:1, :] = agg
            data_4d[:, 0, 2:10, :] = time_feat
            data_4d[:, 1, 0, :] = power[:, 0, :]
            data_4d[:, 1, 1, :] = state[:, 0, :]
            return data_4d
        
        data_train = reconstruct_4d(train_agg, train_time, train_power, train_state)
        data_valid = reconstruct_4d(valid_agg, valid_time, valid_power, valid_state)
        data_test = reconstruct_4d(test_agg, test_time, test_power, test_state)
        data = np.concatenate([data_train, data_valid, data_test], axis=0)
        
        # Create st_date in EXACT format as data_builder returns it
        # get_nilm_dataset returns: (data, (pd.DatetimeIndex,))
        import pandas as pd
        st_date_train = (pd.date_range('2013-01-01', periods=len(data_train), freq='10s'),)
        st_date_valid = (pd.date_range('2013-01-01', periods=len(data_valid), freq='10s'),)
        st_date_test = (pd.date_range('2013-01-01', periods=len(data_test), freq='10s'),)
        st_date = (pd.date_range('2013-01-01', periods=len(data), freq='10s'),)
        
        # Set window size manually since we don't have data_builder
        expes_config.window_size = 256

    elif expes_config.dataset == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path=f"{expes_config.data_path}/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config.app,
            sampling_rate=expes_config.sampling_rate,
            window_size=expes_config.window_size,
        )

        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config.house_with_app_i
        )

        if isinstance(expes_config.window_size, str):
            expes_config.window_size = data_builder.window_size

        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config.seed
            )
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config.seed
            )
        )

    logging.info("             ... Done.")

    scaler = NILMscaler(
        power_scaling_type=expes_config.power_scaling_type,
        appliance_scaling_type=expes_config.appliance_scaling_type,
    )
    data = scaler.fit_transform(data)

    expes_config.cutoff = float(scaler.appliance_stat2[0])
    expes_config.threshold = data_builder.appliance_param[expes_config.app][
        "min_threshold"
    ]

    if expes_config.name_model in ["ConvNet", "ResNet", "Inception"]:
        X, y = nilmdataset_to_tser(data)

        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)

        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    else:
        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

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


def main(dataset, sampling_rate, window_size, appliance, name_model, seed):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (UKDALE or REFIT).
        sampling_rate (int): Selected sampling rate.
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance.
        name_model (str): Name of the model to use for the experiment.
        seed (int): Random seed for reproducibility.
    """

    # Attempt to convert window_size to int
    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    # Load configurations
    with open("configs/expes.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)

        # Dataset name check
        if dataset in datasets_config:
            datasets_config = datasets_config[dataset]
        else:
            raise ValueError(
                "Dataset {} unknown. Only 'UKDALE' and 'REFIT' available.".format(
                    dataset
                )
            )

    with open("configs/models.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        # Selected baseline check
        if name_model in baselines_config:
            expes_config.update(baselines_config[name_model])
        else:
            raise ValueError(
                "Model {} unknown. List of implemented baselines: {}".format(
                    name_model, list(baselines_config.keys())
                )
            )

    # Selected appliance check
    if appliance in datasets_config:
        expes_config.update(datasets_config[appliance])
    else:
        logging.error("Appliance '%s' not found in datasets_config.", appliance)
        raise ValueError(
            "Appliance {} unknown. List of available appliances (for selected {} dataset): {}, ".format(
                appliance, dataset, list(datasets_config.keys())
            )
        )

    # Display experiment config with passed parameters
    logging.info("---- Run experiments with provided parameters ----")
    logging.info("      Dataset: %s", dataset)
    logging.info("      Sampling Rate: %s", sampling_rate)
    logging.info("      Window Size: %s", window_size)
    logging.info("      Appliance : %s", appliance)
    logging.info("      Model: %s", name_model)
    logging.info("      Seed: %s", seed)
    logging.info("--------------------------------------------------")

    # Update experiment config with passed parameters
    expes_config["dataset"] = dataset
    expes_config["appliance"] = appliance
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = name_model

    # Create directories for results
    result_path = create_dir(expes_config["result_path"])
    result_path = create_dir(f"{result_path}{dataset}_{appliance}_{sampling_rate}/")
    result_path = create_dir(f"{result_path}{window_size}/")

    # Cast to OmegaConf
    expes_config = OmegaConf.create(expes_config)

    # Define the path to save experiment results
    expes_config.result_path = (
        f"{result_path}{expes_config.name_model}_{expes_config.seed}"
    )

    # Launch experiments
    launch_one_experiment(expes_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NILMFormer Experiments.")
    parser.add_argument(
        "--dataset", required=True, type=str, help="Dataset name (UKDALE or REFIT)."
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate, e.g. '30s', '1min', '10min', etc.).",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help="Selected appliance, e.g., 'WashingMachine'.",
    )
    parser.add_argument(
        "--name_model", required=True, type=str, help="Name of the model for training."
    )
    parser.add_argument(
        "--seed", required=True, type=int, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        seed=args.seed,
    )
