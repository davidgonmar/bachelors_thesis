#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def _to_millions(x):
    if isinstance(x, dict):
        x = x.get("total", next(iter(x.values())))
    return float(x) / 1e6


def plot_and_save(json_path: str):
    jpath = Path(json_path)
    if not jpath.exists():
        raise FileNotFoundError(jpath)

    with jpath.open() as f:
        data = json.load(f)

    sweep = [d for d in data if "ratio" in d]
    if not sweep:
        raise ValueError("No sweep points with a 'ratio' key found in JSON.")

    metric = sweep[0].get("metric_name", "energy")
    x = [pt["ratio"] for pt in sweep]
    acc = [pt["accuracy"] * (100 if pt["accuracy"] <= 1 else 1) for pt in sweep]
    param = [pt["param_ratio"] * 100 for pt in sweep]
    flops = [_to_millions(pt["flops"]) for pt in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(x, acc, "o-", color="black", lw=2, ms=6, label="Accuracy (%)")
    ax1.plot(
        x, param, "s--", color="darkorange", lw=2, ms=6, label="Parameter ratio (%)"
    )
    ax1.set_xlabel(f"{metric.capitalize()} ratio", fontsize=12)
    ax1.set_ylabel("Accuracy / Parameter ratio (%)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(x, flops, "^-.", color="steelblue", lw=2, ms=6, label="Total FLOPs (M)")
    ax2.set_ylabel("Total FLOPs (millions)", fontsize=12)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=False)

    plt.title(
        f"{metric.capitalize()} sweep: accuracy, parameter growth, compute cost",
        fontsize=13,
    )
    plt.tight_layout()

    pdf_path = jpath.with_suffix(".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot compression sweep (PDF).")
    parser.add_argument("--json_path", help="Path to the JSON results file.")
    args = parser.parse_args()
    plot_and_save(args.json_path)
