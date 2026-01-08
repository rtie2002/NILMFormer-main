# Run all UKDALE experiments with run_one_direct.py (Tensor-based pipeline)
# This script runs training for all appliances using pre-prepared tensors

# List of UKDALE appliances
$appliances = @(
    "Dishwasher",
    "Fridge", 
    "Kettle",
    "Microwave",
    "WashingMachine"
)

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "UKDALE Experiments - NILMFormer (Tensor Pipeline)" -ForegroundColor Green
Write-Host "Total experiments: $($appliances.Count)" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

$experimentCount = 0
$totalExperiments = $appliances.Count
$successCount = 0
$failCount = 0

foreach ($appliance in $appliances) {
    $experimentCount++
    
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Experiment $experimentCount/$totalExperiments" -ForegroundColor Yellow
    Write-Host "Appliance: $appliance" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
    
    # Check if prepared tensors exist
    $tensorDir = "prepared_data/tensors/$($appliance.ToLower())"
    
    if (Test-Path $tensorDir) {
        Write-Host "[OK] Found prepared tensors in $tensorDir" -ForegroundColor Green
        
        # Run training with run_one_direct.py
        Write-Host "Running: python scripts/run_one_direct.py --dataset UKDALE --sampling_rate 1min --window_size 256 --appliance $appliance --name_model NILMFormer --seed 0`n" -ForegroundColor Cyan
        $startTime = Get-Date
        
        python scripts/run_one_direct.py `
            --dataset UKDALE `
            --sampling_rate 1min `
            --window_size 256 `
            --appliance $appliance `
            --name_model NILMFormer `
            --seed 0
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[SUCCESS] Completed $appliance in $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            $successCount++
        }
        else {
            Write-Host "`n[FAILED] Training failed for $appliance" -ForegroundColor Red
            $failCount++
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
        Write-Host "You need to run: python prepared_data/convert_csv_to_pt.py --appliance $($appliance.ToLower())" -ForegroundColor Yellow
        $failCount++
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
Write-Host "Success: $successCount | Failed: $failCount | Total: $totalExperiments" -ForegroundColor Green
Write-Host "Results saved in result/ directory" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

Write-Host "`nResults summary:" -ForegroundColor Cyan
Write-Host "  - Check result/UKDALE_{appliance}_1min/256/NILMFormer_0/ for each appliance" -ForegroundColor White
Write-Host "  - Each directory contains metrics and model checkpoints" -ForegroundColor White
