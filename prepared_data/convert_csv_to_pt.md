# CSV to Tensor Conversion Guide

## Overview

`convert_csv_to_pt.py` converts preprocessed CSV files into PyTorch tensor format (`.pt` files) for faster loading during training.

## Location
```
prepared_data/convert_csv_to_pt.py
```

## What It Does

1. Loads CSV files from `prepared_data/`
2. Calculates global max from ALL splits (train+valid+test) for normalization
3. Applies MaxScaling normalization: `normalized = value / max`
4. Creates sliding windows (default 256 timesteps)
5. Computes appliance states using Kelly paper parameters
6. Saves normalized tensors to `prepared_data/tensors/{appliance}/`

## Output Files

For each appliance, creates:
- `{split}_agg.pt` - Aggregate power (N, 1, 256)
- `{split}_power.pt` - Appliance power (N, 1, 256)
- `{split}_state.pt` - ON/OFF states (N, 1, 256)
- `{split}_time.pt` - Time features (N, 8, 256)
- `stats.pt` - Normalization stats (agg_max, app_max)

Where `{split}` = train, valid, test

---

## Commands for All UKDALE Appliances

### Dishwasher
```powershell
python prepared_data/convert_csv_to_pt.py --appliance dishwasher
```

### Fridge
```powershell
python prepared_data/convert_csv_to_pt.py --appliance fridge
```

### Kettle
```powershell
python prepared_data/convert_csv_to_pt.py --appliance kettle
```

### Microwave
```powershell
python prepared_data/convert_csv_to_pt.py --appliance microwave
```

### Washing Machine
```powershell
python prepared_data/convert_csv_to_pt.py --appliance washingmachine
```

---

## Custom Window Size

Default window size is 256. To use a different size:

```powershell
# Window size 480 (as used in paper for some experiments)
python prepared_data/convert_csv_to_pt.py --appliance dishwasher --window_size 480

# Window size 128
python prepared_data/convert_csv_to_pt.py --appliance fridge --window_size 128
```

---

## Batch Conversion (All Appliances)

PowerShell script to convert all appliances at once:

```powershell
# Convert all UKDALE appliances with window size 256
$appliances = @("dishwasher", "fridge", "kettle", "microwave", "washingmachine")

foreach ($app in $appliances) {
    Write-Host "Converting $app..." -ForegroundColor Cyan
    python prepared_data/convert_csv_to_pt.py --appliance $app
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $app completed" -ForegroundColor Green
    } else {
        Write-Host "✗ $app failed" -ForegroundColor Red
    }
}
```

---

## Expected CSV Files

Script expects these files in `prepared_data/`:
- `{appliance}_training__realPower.csv`
- `{appliance}_validation__realPower.csv`
- `{appliance}_test__realPower.csv`

**CSV columns required:**
- `aggregate` - Total household power
- `{appliance}` - Individual appliance power (e.g., `dishwasher`)
- `minute_sin`, `minute_cos` - Minute of hour (cyclical)
- `hour_sin`, `hour_cos` - Hour of day (cyclical)
- `dow_sin`, `dow_cos` - Day of week (cyclical)
- `month_sin`, `month_cos` - Month of year (cyclical)

- `month_sin`, `month_cos` - Month of year (cyclical)

---

## Validation Strategy (Baseline Replication)

**Recommendation:** To match the baseline results (e.g. 0.0001 loss), you do **NOT** need to manually prepare a validation file.

1.  Put **ALL** your training data into `*_training__realPower.csv`.
2.  Convert it to `train_agg.pt`, `train_time.pt`, etc.
3.  Let the training script (`run_one_direct.py`) automatically split this data (e.g., 80% Train, 20% Valid).

This ensures "In-Distribution" validation, which is easier and consistent with the original paper's methodology. Manual validation files (`valid_*.pt`) are treated as "Out-of-Distribution" (seen as unseen houses) and will result in higher loss.

---

## Normalization Details

**Key Feature (Fixed):**
- Stats calculated from **ALL data** (train+valid+test combined)
- This matches `run_one_expe.py` behavior exactly
- Ensures consistent normalization across splits

**Formula:**
```python
normalized_value = raw_value / global_max
```

**Ranges:**
- Input: 0 to max_power (Watts)
- Output: 0.0 to 1.0 (normalized)

---

## State Computation (Kelly Paper Parameters)

The script applies strict filtering to compute ON/OFF states:

| Appliance | min_on_duration | min_off_duration | min_activation_time |
|-----------|----------------|------------------|---------------------|
| Dishwasher | 180s | 180s | 12s |
| Washing Machine | 180s | 60s | 18s |
| Microwave | 2s | 2s | 2s |
| Kettle | 2s | 0s | 2s |
| Fridge | 2s | 2s | 2s |

**Warning:** These strict parameters can significantly reduce ON samples, especially for dishwasher and washing machine!

---

## Verification

After conversion, verify with:

```powershell
# Check tensor files exist
ls prepared_data/tensors/dishwasher/

# Visualize data
python view_tensors_interactive.py --appliance dishwasher --split train

# Check stats
python -c "import torch; stats = torch.load('prepared_data/tensors/dishwasher/stats.pt', weights_only=False); print(f'agg_max: {stats[\"agg_max\"]:.2f}W'); print(f'app_max: {stats[\"app_max\"]:.2f}W')"
```

---

## Troubleshooting

### Error: "CSV file not found"
**Solution:** Check CSV files exist in `prepared_data/` with correct naming:
```powershell
ls prepared_data/*_training__realPower.csv
```

### Error: "Column not found"
**Solution:** Verify CSV has required columns (aggregate, appliance, time features)

### Low ON ratio warning
**Solution:** This is expected due to Kelly paper filtering. Check:
- Dishwasher: ~0.75-3% ON
- Washing Machine: ~1-2% ON
- Other appliances: Higher ON ratios

### Stats seem wrong
**Solution:** Regenerate with global stats fix (should calculate from all splits combined)

---

## Quick Reference

| Task | Command |
|------|---------|
| Convert dishwasher | `python prepared_data/convert_csv_to_pt.py --appliance dishwasher` |
| Convert with custom window | `python prepared_data/convert_csv_to_pt.py --appliance fridge --window_size 480` |
| Convert all appliances | Use batch script above |
| Verify output | `python view_tensors_interactive.py --appliance {name} --split train` |
| Check stats | `torch.load('prepared_data/tensors/{name}/stats.pt')` |
