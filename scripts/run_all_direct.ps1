# Run all UKDALE experiments with train_direct.py (Window Size 256 only)
# This script runs training for all appliances shown in the paper's Table 2

# Set epochs to proper value for training
Write-Host "Setting epochs to 100 in configs/expes.yaml..." -ForegroundColor Cyan
$configPath = "configs/expes.yaml"
$config = Get-Content $configPath
$config = $config -replace "epochs: !!int \d+", "epochs: !!int 100"
Set-Content $configPath $config

# List of UKDALE appliances from the paper (Table 2)
$appliances = @(
    "dishwasher",
    "fridge", 
    "kettle",
    "microwave",
    "washing_machine"
)

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "UKDALE Experiments - NILMFormer (Window Size 256)" -ForegroundColor Green
Write-Host "Total experiments: $($appliances.Count)" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

$experimentCount = 0
$totalExperiments = $appliances.Count

foreach ($appliance in $appliances) {
    $experimentCount++
    
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Experiment $experimentCount/$totalExperiments" -ForegroundColor Yellow
    Write-Host "Appliance: $appliance" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    
    # Check if prepared tensors exist
    $tensorDir = "prepared_data/tensors/$appliance"
    
    if (Test-Path $tensorDir) {
        Write-Host "[OK] Found prepared tensors in $tensorDir" -ForegroundColor Green
        
        # Run training
        Write-Host "Running: python scripts/train_direct.py --appliance $appliance`n" -ForegroundColor Cyan
        $startTime = Get-Date
        
        python scripts/train_direct.py --appliance $appliance
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[SUCCESS] Completed $appliance in $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        }
        else {
            Write-Host "`n[FAILED] Training failed for $appliance" -ForegroundColor Red
            Write-Host "Continue with next experiment? (Y/N)" -ForegroundColor Yellow
            $response = Read-Host
            if ($response -ne "Y" -and $response -ne "y") {
                Write-Host "Stopping experiments..." -ForegroundColor Red
                exit 1
            }
        }
    }
    else {
        Write-Host "[ERROR] Tensors not found in $tensorDir" -ForegroundColor Red
        Write-Host "You need to run: python scripts/convert_csv_to_pt.py --appliance $appliance" -ForegroundColor Yellow
        Write-Host "Skip this experiment? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -ne "Y" -and $response -ne "y") {
            Write-Host "Stopping experiments..." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETED!" -ForegroundColor Green
Write-Host "Results saved in results/ directory" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

# Restore epochs to 3 for quick testing
Write-Host "Restoring epochs to 3 in configs/expes.yaml..." -ForegroundColor Cyan
$config = Get-Content $configPath
$config = $config -replace "epochs: !!int \d+", "epochs: !!int 3"
Set-Content $configPath $config

Write-Host "`nResults summary:" -ForegroundColor Cyan
Write-Host "  - Check results/{appliance}.pt for each appliance" -ForegroundColor White
Write-Host "  - Each .pt file contains all metrics (timestamp, window, D/W/M)" -ForegroundColor White
Write-Host "`nTo view metrics from a saved file:" -ForegroundColor Cyan
Write-Host '  python -c "import torch; log=torch.load(''results/dishwasher.pt''); print(log[''test_metrics_timestamp''])"' -ForegroundColor White
