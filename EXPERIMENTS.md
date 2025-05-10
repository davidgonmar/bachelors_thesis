# This file contains code to reproduce the experiments for my Bachelor's thesis.

**Assumes resnet20.pth is in the current directory**

## Quantization

### Baseline experiments
First run the baseline experiments.
```
python -m tfg.experiments.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet20 --pretrained_path resnet20.pth
python -m tfg.experiments.quantization.quantize_online --model_name resnet20 --pretrained_path resnet20.pth
python -m tfg.experiments.quantization.quantize_offline --leave_edge_layers_8_bits --model_name resnet20 --pretrained_path resnet20.pth
python -m tfg.experiments.quantization.quantize_offline --model_name resnet20 --pretrained_path resnet20.pth
```
In one line:
```
python -m tfg.experiments.quantization.quantize_online --leave_edge_layers_8_bits --model_name resnet20 --pretrained_path resnet20.pth && python -m tfg.experiments.quantization.quantize_online --model_name resnet20 --pretrained_path resnet20.pth && python -m tfg.experiments.quantization.quantize_offline --leave_edge_layers_8_bits --model_name resnet20 --pretrained_path resnet20.pth && python -m tfg.experiments.quantization.quantize_offline --model_name resnet20 --pretrained_path resnet20.pth
```

To generate the heatmaps:
```
python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_offline_resnet20_True.json
python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_offline_resnet20_False.json
python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_online_resnet20_True.json
python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_online_resnet20_False.json
```
In one line:
```
python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_offline_resnet20_True.json && python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_offline_resnet20_False.json && python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_online_resnet20_True.json && python -m tfg.experiments.quantization.accuracy_heatmap_generator quantization_results_online_resnet20_False.json
```

## Factorization

## Baseline experiments
First run the baseline experiments.
```
python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric energy
python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric rank
python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric params_ratio
```
In one line:
```
python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric energy && python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric rank && python -m tfg.experiments.factorization.svd_uniform --model_name resnet20 --pretrained_path resnet20.pth --metric params_ratio
```

To generate the plots:
```
python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_energy.json
python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_rank.json
python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_params_ratio.json
```
In one line:
```
python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_energy.json && python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_rank.json && python -m tfg.experiments.factorization.plot_factorization --json_path factorization_results_resnet20_params_ratio.json
```