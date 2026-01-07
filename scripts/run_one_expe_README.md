# run_one_expe.py - Usage Guide

## Overview

`run_one_expe.py` is the **original NILMFormer training script** that processes raw UK-DALE data directly from `.dat` files. This is the reference implementation that produces baseline results.

## Data Source

- **Input:** Raw UK-DALE `.dat` files from `data/UKDALE/House1/`, `House2/`, etc.
- **Processing:** Loads raw data → Normalizes → Creates windows → Trains model
- **Speed:** Slower (preprocessing on-the-fly)
- **Accuracy:** ✅ Baseline reference results

## Prerequisites

**You must have raw UK-DALE dataset:**
```
data/UKDALE/
├── House1/
│   ├── channel_1.dat  (aggregate)
│   ├── channel_5.dat  (fridge)
│   ├── ...
├── House2/
├── House3/
├── House4/
└── House5/
```

If you don't have raw UK-DALE data, use `run_one_direct.py` instead!

---

## Usage Commands

### Basic Command

```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0
```

### All UKDALE Appliances

**Dishwasher:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0
```

**Fridge:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Fridge --name_model NILMFormer --seed 0
```

**Kettle:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Kettle --name_model NILMFormer --seed 0
```

**Microwave:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Microwave --name_model NILMFormer --seed 0
```

**Washing Machine:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance WashingMachine --name_model NILMFormer --seed 0
```

---

## Window Sizes (from Paper)

The NILMFormer paper tested multiple window sizes:

**Window 256:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance Dishwasher --name_model NILMFormer --seed 0
```

**Window 480:**
```powershell
python scripts/run_one_expe.py --dataset UKDALE --sampling_rate 1min --window_size 480 --appliance Dishwasher --name_model NILMFormer --seed 0
```

---

## Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dataset` | Dataset name | `UKDALE` or `REFIT` |
| `--sampling_rate` | Data sampling rate | `1min` (10 seconds per sample) |
| `--window_size` | Window length | `256`, `480`, or `day`/`week`/`month` |
| `--appliance` | Target appliance (capitalized) | `Dishwasher`, `Fridge`, etc. |
| `--name_model` | Model architecture | `NILMFormer`, `Seq2Point`, `Seq2Seq`, etc. |
| `--seed` | Random seed for reproducibility | `0` |

---

## How It Works

1. **Loads raw UK-DALE data** from `.dat` files
2. **Applies preprocessing:**
   - Resamples to 1-minute intervals
   - Computes appliance states using Kelly paper parameters
   - Creates sliding windows
3. **Splits data:**
   - Train: Houses [1, 3, 4, 5]
   - Valid: 20% of training houses
   - Test: House [2]
4. **Normalizes:** MaxScaling on ALL data (train+valid+test combined)
5. **Trains model** using SeqToSeqTrainer
6. **Evaluates:**
   - Timestamp-level metrics (MAE, F1-score)
   - Window-level aggregation
   - Daily/Weekly/Monthly energy aggregation

---

## House Splits (from configs/datasets.yaml)

**Dishwasher:**
- Train: [1, 3, 4, 5]
- Test: [2]

**Fridge:**
- Train: [1, 2, 3, 4]
- Test: [5]

**Kettle:**
- Train: [1, 3, 4, 5]
- Test: [2]

**Microwave:**
- Train: [1, 2, 3, 4, 5] 
- Test: [2]

**Washing Machine:**
- Train: [1, 3, 5]
- Test: [2]

---

## Expected Results (from Paper - Window 256)

| Appliance | MAE (W) | F1-Score |
|-----------|---------|----------|
| Dishwasher | ~16.7 | ~0.79 |
| Fridge | ~13.8 | ~0.81 |
| Kettle | ~25.9 | ~0.85 |
| Microwave | ~8.9 | ~0.82 |
| Washing Machine | ~18.3 | ~0.72 |

---

## Troubleshooting

### Error: "No such file or directory: data/UKDALE/"
**Solution:** You don't have raw UK-DALE data. Either:
1. Download UK-DALE dataset
2. OR use `run_one_direct.py` with pre-prepared tensors

### Error: "Appliance unknown"
**Solution:** Use capitalized names:
- ✅ `Dishwasher`, `Fridge`, `Kettle`, `Microwave`, `WashingMachine`
- ❌ `dishwasher`, `washing_machine`

### Slow execution
**Solution:** Normal - processing raw data is slow. Use `run_one_direct.py` for faster training with pre-prepared data.

### Different results from paper
**Possible causes:**
1. Different UK-DALE version/preprocessing
2. Different random seed
3. Different number of epochs

---

## Comparison with run_one_direct.py

| Feature | run_one_expe.py | run_one_direct.py |
|---------|----------------|-------------------|
| **Data Source** | Raw `.dat` files | Pre-prepared `.pt` tensors |
| **Speed** | Slower | Faster |
| **Setup Required** | Raw UK-DALE download | CSV → tensor conversion |
| **Pipeline** | Full from raw data | Same, starting from normalized data |
| **Results** | ✅ Baseline reference | Should match if data source is same |

**Recommendation:**
- Use `run_one_expe.py` if you have raw UK-DALE data
- Use `run_one_direct.py` if you only have CSV files or want faster training
