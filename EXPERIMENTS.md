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