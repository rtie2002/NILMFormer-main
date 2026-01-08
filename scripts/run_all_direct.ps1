# Global parameters
$SEEDS = @(0)

$DATASETS = @("UKDALE")
$APPLIANCES = @("washing_machine", "dishwasher", "kettle", "microwave", "fridge")

$MODELS = @("NILMFormer")
$WINDOW_SIZES = @("256")

# Run experiments
function Run-Batch {
    param (
        [Parameter(Mandatory = $true)] [string[]]$Datasets,
        [Parameter(Mandatory = $true)] [string[]]$Appliances,
        [Parameter(Mandatory = $true)] [string[]]$Models,
        [Parameter(Mandatory = $true)] [string[]]$WindowSizes
    )

    foreach ($dataset in $Datasets) {
        foreach ($appliance in $Appliances) {
            foreach ($win in $WindowSizes) {
                foreach ($model in $Models) {
                    foreach ($seed in $SEEDS) {
                        Write-Host "`n========================================" -ForegroundColor Cyan
                        Write-Host "Running: python scripts\run_one_direct.py --dataset $dataset --sampling_rate 1min --appliance $appliance --window_size $win --name_model $model --seed $seed" -ForegroundColor Yellow
                        Write-Host "========================================`n" -ForegroundColor Cyan
                        
                        python scripts\run_one_direct.py `
                            --dataset "$dataset" `
                            --sampling_rate "1min" `
                            --appliance "$appliance" `
                            --window_size "$win" `
                            --name_model "$model" `
                            --seed "$seed"
                        
                        # Display evaluation results
                        $resultPath = "result/${dataset}_${appliance}_1min/${win}/${model}_${seed}.pt"
                        if (Test-Path $resultPath) {
                            Write-Host "`n========================================" -ForegroundColor Green
                            Write-Host "Evaluation Results for $dataset - $appliance - $model (seed $seed)" -ForegroundColor Green
                            Write-Host "========================================" -ForegroundColor Green
                            
                            # Use Python to read and display the metrics from the .pt file
                            python -c @"
import torch
import sys

try:
    log = torch.load('$resultPath', weights_only=False)
    
    print('\n--- Test Metrics (Timestamp) ---')
    if 'test_metrics_timestamp' in log:
        metrics = log['test_metrics_timestamp']
        for key, value in metrics.items():
            print(f'  {key}: {value:.6f}')
    
    print('\n--- Test Metrics (Window) ---')
    if 'test_metrics_win' in log:
        metrics = log['test_metrics_win']
        for key, value in metrics.items():
            print(f'  {key}: {value:.6f}')
    
    print('\n--- Validation Metrics (Timestamp) ---')
    if 'valid_metrics_timestamp' in log:
        metrics = log['valid_metrics_timestamp']
        for key, value in metrics.items():
            print(f'  {key}: {value:.6f}')
    
    print('\n--- Training Info ---')
    if 'epoch_best_loss' in log:
        print(f'  Best Epoch: {log[\"epoch_best_loss\"]}')
    if 'value_best_loss' in log:
        print(f'  Best Loss: {log[\"value_best_loss\"]:.6f}')
    if 'training_time' in log:
        print(f'  Training Time: {log[\"training_time\"]:.2f}s')
    if 'test_metrics_time' in log:
        print(f'  Test Time: {log[\"test_metrics_time\"]:.2f}s')
        
except Exception as e:
    print(f'Error reading results: {e}', file=sys.stderr)
"@
                            Write-Host "========================================`n" -ForegroundColor Green
                        }
                        else {
                            Write-Host "`nWarning: Result file not found at $resultPath`n" -ForegroundColor Red
                        }
                    }
                }
            }
        }
    }
}

#####################################
# Run all possible experiments
#####################################
Run-Batch -Datasets $DATASETS -Appliances $APPLIANCES -Models $MODELS -WindowSizes $WINDOW_SIZES
Write-Host "`nDetailed results saved in:" -ForegroundColor Cyan
Write-Host "  result/UKDALE_{appliance}_1min/256/NILMFormer_{seed}/" -ForegroundColor White

# Summarize Results
python scripts\summarize_results.py
