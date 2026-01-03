# Global parameters
$SEEDS = @(0, 1, 2)

$DATASETS_1 = @("REFIT")
$APPLIANCES_1 = @("WashingMachine", "Dishwasher", "Kettle", "Microwave")

$DATASETS_2 = @("UKDALE")
$APPLIANCES_2 = @("WashingMachine", "Dishwasher", "Kettle", "Microwave", "Fridge")

$MODELS_1 = @("BiLSTM", "FCN", "CNN1D", "UNetNILM", "DAResNet", "BERT4NILM", "DiffNILM", `
        "TSILNet", "Energformer", "BiGRU", "STNILM", "NILMFormer")
$WINDOW_SIZES_1 = @("128", "256", "512", "360", "720")

$MODELS_2 = @("ConvNet", "ResNet", "Inception")
$WINDOW_SIZES_2 = @("day", "week", "month")

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
                        Write-Host "Running: python -m scripts.run_one_expe --dataset $dataset --sampling_rate 1min --appliance $appliance --window_size $win --name_model $model --seed $seed"
                        
                        python -m scripts.run_one_expe `
                            --dataset "$dataset" `
                            --sampling_rate "1min" `
                            --appliance "$appliance" `
                            --window_size "$win" `
                            --name_model "$model" `
                            --seed "$seed"
                    }
                }
            }
        }
    }
}

#####################################
# Run all possible experiments
#####################################
Run-Batch -Datasets $DATASETS_1 -Appliances $APPLIANCES_1 -Models $MODELS_1 -WindowSizes $WINDOW_SIZES_1
Run-Batch -Datasets $DATASETS_1 -Appliances $APPLIANCES_1 -Models $MODELS_2 -WindowSizes $WINDOW_SIZES_2
Run-Batch -Datasets $DATASETS_2 -Appliances $APPLIANCES_2 -Models $MODELS_1 -WindowSizes $WINDOW_SIZES_1
Run-Batch -Datasets $DATASETS_2 -Appliances $APPLIANCES_2 -Models $MODELS_2 -WindowSizes $WINDOW_SIZES_2
