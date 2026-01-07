# Tensor Data Visualization Guide

## Overview

There are two visualization tools available:
1. **`visualize_tensors.py`** - Creates static PNG images with multiple windows
2. **`view_tensors_interactive.py`** - Interactive viewer to navigate window-by-window

## Setup (Run Once Per Session)

```powershell
# Navigate to project root
cd C:\Users\Raymond Tie\Desktop\PhD\Code\NILMFormer-main

# Fix OpenMP library conflict
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

---

## Option 1: Interactive Viewer (RECOMMENDED)

**Best for:** Browsing through windows one at a time, inspecting individual samples

### Command
```powershell
python view_tensors_interactive.py --appliance dishwasher --split train
```

### Features
- ✅ Shows ONE window at a time (clean view)
- ✅ Navigate with **"Next →"** and **"← Previous"** buttons
- ✅ Or use **arrow keys** (← →) on keyboard
- ✅ Press **ESC** or **Q** to close
- ✅ No files saved (live viewing only)
- ✅ Window info displayed: number, average power, ON ratio

### What You See
**4 plots per window:**
1. **Top-left**: Aggregate power (total household consumption)
2. **Top-right**: Appliance power (dishwasher only)
3. **Bottom-left**: State labels (green = ON, gray = OFF)
4. **Bottom-right**: Time features (minute, hour, day-of-week cyclical encodings)

### Examples

**Dishwasher:**
```powershell
python view_tensors_interactive.py --appliance dishwasher --split train
python view_tensors_interactive.py --appliance dishwasher --split valid
python view_tensors_interactive.py --appliance dishwasher --split test
```

**Fridge:**
```powershell
python view_tensors_interactive.py --appliance fridge --split train
python view_tensors_interactive.py --appliance fridge --split valid
python view_tensors_interactive.py --appliance fridge --split test
```

**Kettle:**
```powershell
python view_tensors_interactive.py --appliance kettle --split train
python view_tensors_interactive.py --appliance kettle --split valid
python view_tensors_interactive.py --appliance kettle --split test
```

**Microwave:**
```powershell
python view_tensors_interactive.py --appliance microwave --split train
python view_tensors_interactive.py --appliance microwave --split valid
python view_tensors_interactive.py --appliance microwave --split test
```

**Washing Machine:**
```powershell
python view_tensors_interactive.py --appliance washingmachine --split train
python view_tensors_interactive.py --appliance washingmachine --split valid
python view_tensors_interactive.py --appliance washingmachine --split test
```

---

## Option 2: Static Visualization

**Best for:** Creating reports, documentation, or viewing multiple windows at once

### Command
```powershell
python visualize_tensors.py --appliance dishwasher --split train
```

### Features
- ✅ Shows **5 windows** (default) or more side-by-side
- ✅ Saves **PNG files** for documentation
- ✅ Includes **distribution plots** (histograms, scatter plots)
- ✅ Good for comparing multiple windows

### Options
```powershell
# Show more windows (e.g., 10)
python visualize_tensors.py --appliance dishwasher --split train --num_windows 10

# Different splits
python visualize_tensors.py --appliance dishwasher --split valid
python visualize_tensors.py --appliance dishwasher --split test
```

### Output Files
Creates 2 PNG files in project root:
1. `tensor_visualization_dishwasher_train.png` - Time series plots (5+ windows stacked)
2. `tensor_distributions_dishwasher_train.png` - 4 distribution plots:
   - Appliance power histogram
   - State ON/OFF bar chart
   - Aggregate vs Appliance scatter plot
   - Non-zero power distribution

---

## Understanding the Plots

### Aggregate Power
- **Blue line**: Total household electricity consumption
- **Range**: 0 to ~7920W (max seen in data)
- **What it shows**: All appliances + base load combined

### Appliance Power
- **Red line**: Individual appliance consumption (dishwasher)
- **Range**: 0 to ~2551W (max for dishwasher)
- **What it shows**: Ground truth power for the target appliance

### State (ON/OFF)
- **Green fill**: Appliance is ON (state = 1)
- **White/Gray**: Appliance is OFF (state = 0)
- **What it shows**: Classification labels after Kelly paper filtering
  - Dishwasher: Only ~0.75-2.96% ON due to strict filtering

### Time Features
- **Cyan**: `minute_sin` - Time of hour (cyclical)
- **Magenta**: `hour_sin` - Hour of day (cyclical)
- **Yellow**: `dow_sin` - Day of week (cyclical)
- **Range**: -1 to +1 (sine/cosine encoding)
- **What it shows**: Temporal patterns that help the model learn appliance usage schedules

---

## Data Statistics

**For dishwasher (window_size=256):**
- Training: 3,932 windows
- Validation: 983 windows  
- Test: 674 windows
- Each window: 256 timesteps (10-second intervals = 42.6 minutes)

**Normalization:**
- Aggregate: Divided by `agg_max` (≈7920W)
- Appliance: Divided by `app_max` (≈2551W)
- Time features: Already in [-1, 1] range

---

## Troubleshooting

### Error: "No such file or directory"
**Solution:** Make sure you're in the project root:
```powershell
cd C:\Users\Raymond Tie\Desktop\PhD\Code\NILMFormer-main
```

### Error: "OMP: Error #15"
**Solution:** Run the setup command:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### Error: "can't open file"
**Solution:** Make sure the tensor files exist:
```powershell
ls prepared_data/tensors/dishwasher/
```
Should show: `train_agg.pt`, `train_power.pt`, `train_state.pt`, `train_time.pt`, `stats.pt`

### Window doesn't show
**Solution:** The window might be behind other windows - check taskbar

---

## Quick Reference

| Task | Command |
|------|---------|
| Browse windows interactively | `python view_tensors_interactive.py --appliance dishwasher --split train` |
| Generate report images | `python visualize_tensors.py --appliance dishwasher --split train` |
| View validation data | Add `--split valid` to either command |
| View test data | Add `--split test` to either command |
| Show more windows | Add `--num_windows 10` to `visualize_tensors.py` |

---

## Tips

1. **Use interactive viewer** for detailed inspection of individual windows
2. **Use static viewer** when you need to compare multiple windows side-by-side
3. **Check state ON ratio** - if too low (<1%), model may struggle to learn
4. **Look for patterns** - Does appliance power correlate with certain times?
5. **Verify normalization** - Power values should be in [0, 1] range in the tensors
