# Global parameters
$SEEDS = @(0)

$DATASETS = @("UKDALE")
$APPLIANCES = @("WashingMachine", "Dishwasher", "Kettle", "Microwave", "Fridge")

$MODELS = @("NILMFormer")
$WINDOW_SIZES = @("128", "256", "512")
$EPOCH_CONFIGS = @(3, 10)  # Run experiments with both 3 and 10 epochs

# Run experiments
function Run-Batch {
    param (
        [Parameter(Mandatory = $true)] [string[]]$Datasets,
        [Parameter(Mandatory = $true)] [string[]]$Appliances,
        [Parameter(Mandatory = $true)] [string[]]$Models,
        [Parameter(Mandatory = $true)] [string[]]$WindowSizes,
        [Parameter(Mandatory = $true)] [int[]]$EpochConfigs
    )

    foreach ($dataset in $Datasets) {
        foreach ($appliance in $Appliances) {
            foreach ($win in $WindowSizes) {
                foreach ($model in $Models) {
                    foreach ($epochs in $EpochConfigs) {
                        foreach ($seed in $SEEDS) {
                            Write-Host "`n========================================" -ForegroundColor Cyan
                            Write-Host "Running: $dataset - $appliance - Window $win - $epochs epochs - Seed $seed" -ForegroundColor Yellow
                            Write-Host "========================================`n" -ForegroundColor Cyan
                            
                            # Update expes.yaml to set the number of epochs
                            $configPath = "configs\expes.yaml"
                            $configContent = Get-Content $configPath -Raw
                            $configContent = $configContent -replace "epochs: !!int \d+", "epochs: !!int $epochs"
                            Set-Content -Path $configPath -Value $configContent
                            
                            Write-Host "Updated config: epochs = $epochs" -ForegroundColor Green
                            
                            # Run the experiment
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
                                Write-Host "Results: $dataset - $appliance - Window $win - $epochs epochs (seed $seed)" -ForegroundColor Green
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
    
    print('\n--- Training Info ---')
    if 'epoch_best_loss' in log:
        print(f'  Best Epoch: {log["epoch_best_loss"]}')
    if 'value_best_loss' in log:
        print(f'  Best Loss: {log["value_best_loss"]:.6f}')
    if 'training_time' in log:
        print(f'  Training Time: {log["training_time"]:.2f}s')
        
except Exception as e:
    print(f'Error reading results: {e}', file=sys.stderr)
"@
                                Write-Host "========================================`n" -ForegroundColor Green
                                
                                # Rename result file to include epoch count
                                $newResultPath = "result/${dataset}_${appliance}_1min/${win}/${model}_${seed}_${epochs}ep.pt"
                                Move-Item -Path $resultPath -Destination $newResultPath -Force
                                Write-Host "Saved results to: $newResultPath" -ForegroundColor Magenta
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
}

#####################################
# Run all possible experiments
#####################################
Run-Batch -Datasets $DATASETS -Appliances $APPLIANCES -Models $MODELS -WindowSizes $WINDOW_SIZES -EpochConfigs $EPOCH_CONFIGS

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "       ALL EXPERIMENTS COMPLETED       " -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

Write-Host "`nResults saved with naming convention:" -ForegroundColor Cyan
Write-Host "  result/UKDALE_{appliance}_1min/{window}/NILMFormer_{seed}_{epochs}ep.pt" -ForegroundColor White
Write-Host "`nExample:" -ForegroundColor Cyan
Write-Host "  - NILMFormer_0_3ep.pt  (3 epochs)" -ForegroundColor White
Write-Host "  - NILMFormer_0_10ep.pt (10 epochs)" -ForegroundColor White

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "       SUMMARY OF RESULTS (3 epochs)   " -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
# Note: summarize_results.py will need to be updated to handle the new naming convention
Write-Host "To compare results, check the individual .pt files" -ForegroundColor Yellow
Write-Host "or update summarize_results.py to handle _3ep and _10ep suffixes" -ForegroundColor Yellow
