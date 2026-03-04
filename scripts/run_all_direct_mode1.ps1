# ============================================================
# run_all_direct_mode1.ps1
# Runs NILMFormer on all Mode1 scenarios.
# Mode1 tensor structure:  prepared_data_Mode1/tensors/{window}/{app}/{scenario}/
# Scenario names come from the CSV filenames, e.g. 200k+10k_ordered
# ============================================================

$SEEDS = @(0)
$DATASETS = @("UKDALE")
$MODELS = @("NILMFormer")
$WINDOW_SIZES = @("256", "128", "512")

# Manual appliance list (Order here = Training Order)
$APPLIANCES = @("WashingMachine", "Microwave", "Dishwasher", "Fridge", "Kettle")

# Set reproducibility env vars
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# ── Helper: list scenario names for one appliance + window ──────────────────
function Get-Scenarios {
    param([string]$App, [string]$Win)
    $tensorBase = "prepared_data_Mode1\tensors\$Win\$($App.ToLower())"
    if (-not (Test-Path $tensorBase)) { return @() }
    return (Get-ChildItem -Directory $tensorBase).Name
}

# ── Main batch loop ─────────────────────────────────────────────────────────
foreach ($dataset in $DATASETS) {
    foreach ($win in $WINDOW_SIZES) {
        foreach ($appliance in $APPLIANCES) {
            $scenarios = Get-Scenarios -App $appliance -Win $win
            if ($scenarios.Count -eq 0) {
                Write-Host "  [SKIP] No scenarios found for $appliance / win=$win" -ForegroundColor Yellow
                continue
            }

            foreach ($scenario in $scenarios) {
                foreach ($model in $MODELS) {
                    foreach ($seed in $SEEDS) {

                        Write-Host "`n========================================" -ForegroundColor Cyan
                        Write-Host "Dataset=$dataset  App=$appliance  Win=$win" -ForegroundColor Yellow
                        Write-Host "Scenario=$scenario  Model=$model  Seed=$seed" -ForegroundColor Yellow
                        Write-Host "========================================`n" -ForegroundColor Cyan

                        $resultPath = "result\mode1\${dataset}_${appliance}_1min_${scenario}\${win}\${model}_${seed}.pt"
                        
                        # Fix: Cleanup old results to prevent 11.44 contamination
                        if (Test-Path $resultPath) {
                            Remove-Item -Force $resultPath
                        }

                        python scripts\run_one_direct_mode1.py `
                            --dataset      "$dataset" `
                            --sampling_rate "1min" `
                            --appliance    "$appliance" `
                            --window_size  "$win" `
                            --name_model   "$model" `
                            --seed         "$seed" `
                            --scenario     "$scenario"

                        # Show result if saved
                        if (Test-Path $resultPath) {
                            $resultPathPy = $resultPath.Replace("\", "/")
                            Write-Host "`n--- Results: $appliance | $scenario | win=$win ---" -ForegroundColor Green
                            python -c @"
import torch, sys
try:
    log = torch.load('$resultPathPy', weights_only=False)
    print('\nTest Metrics (Timestamp):')
    for k,v in log.get('test_metrics_timestamp', {}).items():
        print(f'  {k}: {v:.4f}')
    print('\nTest Metrics (Window):')
    for k,v in log.get('test_metrics_win', {}).items():
        print(f'  {k}: {v:.4f}')
    if 'epoch_best_loss' in log:
        print(f"\n  Best Epoch : {log['epoch_best_loss']}")
    if 'value_best_loss' in log:
        print(f"  Best Loss  : {log['value_best_loss']:.6f}")
except Exception as e:
    print(f'Error reading result: {e}')
"@
                        }
                        else {
                            Write-Host "  [WARN] Result not found: $resultPath" -ForegroundColor Red
                        }

                        # Update summary table after EACH experiment
                        python scripts\summarize_results_mode1.py
                    }
                }
            }
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "       ALL MODE1 EXPERIMENTS DONE      " -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta
