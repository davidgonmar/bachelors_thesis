import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.quantization import (
    to_quantized_online,
)
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean as mean,
    cifar10_std as std,
)
from compress.quantization.recipes import (
    get_recipe_quant,
)
import argparse
from itertools import product
import json

parser = argparse.ArgumentParser()
parser.add_argument("--leave_edge_layers_8_bits", action="store_true")
parser.add_argument("--model_name", type=str, default=None, required=True)
parser.add_argument("--pretrained_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=512)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

model = load_vision_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.model_name),
    modifier_after_load=None,
    model_args={"num_classes": 10},
)
model.eval()

model.to(device)
# cifar10 mean and std

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

eval_results = evaluate_vision_model(model, test_loader)

results = []
results.append(
    {
        "type": "original",
        "loss": eval_results["loss"],
        "accuracy": eval_results["accuracy"],
    }
)

model.to("cpu")

bit_widths = [2, 4, 8, 16]
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

for w_bits, act_bits in product(bit_widths, bit_widths):

    specs = get_recipe_quant(
        args.model_name,
    )(
        bits_activation=act_bits,
        bits_weight=w_bits,
        clip_percentile=0.995,
        leave_edge_layers_8_bits=args.leave_edge_layers_8_bits,
        symmetric=False,
    )
    quanted = to_quantized_online(
        model.to(device),
        specs,
        inplace=False,
    )
    model.to("cpu")
    quanted.to(device)
    eval_results = evaluate_vision_model(quanted, test_loader)
    print(
        f"Quantized model with {w_bits} bits for weights and {act_bits} bits for activations: "
        f"loss: {eval_results['loss']}, accuracy: {eval_results['accuracy']}"
    )
    results.append(
        {
            "type": f"W{w_bits}A{act_bits}",
            "leave_edge_layers_8_bits": args.leave_edge_layers_8_bits,
            "loss": eval_results["loss"],
            "accuracy": eval_results["accuracy"],
        }
    )
file_name = f"quantization_results_online_{args.model_name}_{args.leave_edge_layers_8_bits}.json"
with open(file_name, "w") as f:
    json.dump(results, f, indent=4)
