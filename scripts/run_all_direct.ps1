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

# Results storage
$results = @()

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
        
        # Capture output to parse metrics
        python scripts/run_one_direct.py `
            --dataset UKDALE `
            --sampling_rate 1min `
            --window_size 256 `
            --appliance $appliance `
            --name_model NILMFormer `
            --seed 0 2>&1 | Tee-Object -Variable capturedOutput | Out-Null
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[SUCCESS] Completed $appliance in $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            $successCount++
            
            # Parse metrics from output
            $testMAE = "N/A"
            $testF1 = "N/A"
            $testAcc = "N/A"
            $validLoss = "N/A"
            
            foreach ($line in $capturedOutput) {
                if ($line -match "Test.*MAE.*:\s*([\d.]+)") {
                    $testMAE = [math]::Round([double]$matches[1], 2)
                }
                if ($line -match "Test.*F1.*:\s*([\d.]+)") {
                    $testF1 = [math]::Round([double]$matches[1], 4)
                }
                if ($line -match "Test.*Accuracy.*:\s*([\d.]+)") {
                    $testAcc = [math]::Round([double]$matches[1], 4)
                }
                if ($line -match "Valid\s+loss\s*:\s*([\d.]+)") {
                    $validLoss = [math]::Round([double]$matches[1], 6)
                }
            }
            
            # Store results
            $results += [PSCustomObject]@{
                Appliance = $appliance
                Status    = "✓"
                ValidLoss = $validLoss
                TestMAE   = $testMAE
                TestF1    = $testF1
                TestAcc   = $testAcc
                Duration  = $duration.ToString('hh\:mm\:ss')
            }
            
            # Display immediate results
            Write-Host ""
            Write-Host "--- Results for $appliance ---" -ForegroundColor Cyan
            Write-Host "  Valid Loss: $validLoss" -ForegroundColor White
            Write-Host "  Test MAE:   $testMAE W" -ForegroundColor White
            Write-Host "  Test F1:    $testF1" -ForegroundColor White
            Write-Host "  Test Acc:   $testAcc" -ForegroundColor White
        }
        else {
            Write-Host "`n[FAILED] Training failed for $appliance" -ForegroundColor Red
            $failCount++
            
            # Store failure
            $results += [PSCustomObject]@{
                Appliance = $appliance
                Status    = "✗"
                ValidLoss = "FAILED"
                TestMAE   = "FAILED"
                TestF1    = "FAILED"
                TestAcc   = "FAILED"
                Duration  = $duration.ToString('hh\:mm\:ss')
            }
            
            Write-Host "Continue with next experiment? (Y/N)" -ForegroundColor Yellow
            $response = Read-Host
            if ($response -ne "Y" -and $response -ne "y") {
                Write-Host "Stopping experiments..." -ForegroundColor Red
                break
            }
        }
    }
    else {
        Write-Host "[ERROR] Tensors not found in $tensorDir" -ForegroundColor Red
        Write-Host "You need to run: python prepared_data/convert_csv_to_pt.py --appliance $($appliance.ToLower())" -ForegroundColor Yellow
        $failCount++
        
        # Store skip
        $results += [PSCustomObject]@{
            Appliance = $appliance
            Status    = "⊘"
            ValidLoss = "SKIPPED"
            TestMAE   = "SKIPPED"
            TestF1    = "SKIPPED"
            TestAcc   = "SKIPPED"
            Duration  = "00:00:00"
        }
        
        Write-Host "Skip this experiment? (Y/N)" -ForegroundColor Yellow
        $response = Read-Host
        if ($response -ne "Y" -and $response -ne "y") {
            Write-Host "Stopping experiments..." -ForegroundColor Red
            break
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETED!" -ForegroundColor Green
Write-Host "Success: $successCount | Failed: $failCount | Total: $totalExperiments" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

# Display results table
Write-Host "`n========== RESULTS SUMMARY TABLE ==========" -ForegroundColor Cyan
$results | Format-Table -AutoSize -Property Appliance, Status, ValidLoss, TestMAE, TestF1, TestAcc, Duration

Write-Host "`nLegend:" -ForegroundColor Yellow
Write-Host "  ✓ = Success  |  ✗ = Failed  |  ⊘ = Skipped (No tensors)" -ForegroundColor White

Write-Host "`nDetailed results saved in:" -ForegroundColor Cyan
Write-Host "  result/UKDALE_{appliance}_1min/256/NILMFormer_0/" -ForegroundColor White
