# run_one_direct.py - Usage Guide

## Overview

`run_one_direct.py` is a modified version of `run_one_expe.py` that uses **pre-prepared tensor data** instead of raw UKDALE files. This guarantees the pipeline is identical to the original, just with faster data loading.

## Key Differences from run_one_expe.py

| Feature | run_one_expe.py | run_one_direct.py |
|---------|----------------|-------------------|
| **Data Source** | Raw `.dat` files from `data/UKDALE/` | Pre-prepared `.pt` tensors from `prepared_data/tensors/` |
| **Preprocessing** | Loads raw → normalizes → windows | Tensors already normalized and windowed |
| **Speed** | Slower (preprocessing on-the-fly) | Faster (data preloaded) |
| **Pipeline** | Full pipeline from raw data | Same pipeline starting from normalized data |
| **Results** | ✅ Baseline reference | ✅ Should match run_one_expe.py exactly |

## Usage

### Basic Command

```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0
```

### All UKDALE Appliances

**Dishwasher:**
```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0
```

**Fridge:**
```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Fridge --name_model NILMFormer --seed 0
```

**Kettle:**
```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Kettle --name_model NILMFormer --seed 0
```

**Microwave:**
```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Microwave --name_model NILMFormer --seed 0
```

**Washing Machine:**
```powershell
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance WashingMachine --name_model NILMFormer --seed 0
```

## Prerequisites

1. **Tensor files must exist** in `prepared_data/tensors/{appliance}/`:
   - `train_agg.pt`, `train_power.pt`, `train_state.pt`, `train_time.pt`
   - `valid_agg.pt`, `valid_power.pt`, `valid_state.pt`, `valid_time.pt`
   - `test_agg.pt`, `test_power.pt`, `test_state.pt`, `test_time.pt`
   - `stats.pt`

2. **Generate tensors first** if they don't exist:
   ```powershell
   python prepared_data/convert_csv_to_pt.py --appliance dishwasher
   ```

## How It Works

1. **Loads tensors** from `prepared_data/tensors/{appliance}/`
2. **Reconstructs 4D arrays** matching `run_one_expe.py` format: `(N, 2, 10, window_size)`
3. **Sets up scaler** with stats from `stats.pt` (for denormalization during evaluation)
4. **Skips normalization** since data is already normalized
5. **Uses identical training pipeline** as `run_one_expe.py`

## Validation

Compare results with `run_one_expe.py` to ensure pipeline equivalence:

```powershell
# Run with tensors
python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0

# Run with raw data (if you have it)
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0

# Compare results
```

Results should be very similar if the CSV files used to create tensors match the raw data.

## Troubleshooting

### Error: "No such file"Unable to find tensors**
**Solution:** Generate tensors first:
```powershell
python prepared_data/convert_csv_to_pt.py --appliance dishwasher
```

### Error: "Appliance unknown"
**Solution:** Use capitalized appliance names:
- ✅ `Dishwasher`, `Fridge`, `Kettle`, `Microwave`, `WashingMachine`
- ❌ `dishwasher`, `washing_machine`

### Different results from run_one_expe.py
**Possible causes:**
1. CSV files use different houses than `configs/datasets.yaml` specifies
2. Different preprocessing was applied to create CSVs
3. Stats calculated differently (should be from ALL data combined)

**Solution:** Ensure your CSVs match the exact preprocessing of `run_one_expe.py`

## Notes

- Appliance names in command are **capitalized** (e.g., `Dishwasher`)
- Tensor folder names are **lowercase** (e.g., `dishwasher/`)
- Data is already normalized - no double normalization occurs
- Scaler is still needed for denormalization during evaluation
