#!/usr/bin/env python3
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    w_bits = [2, 4, 8, 16]
    a_bits = [2, 4, 8, 16]
    acc = np.zeros((len(a_bits), len(w_bits)))

    for entry in data:
        t = entry["type"]
        if t == "original":
            continue
        w = int(t.split("W")[1].split("A")[0])
        a = int(t.split("A")[1])
        i = a_bits.index(a)
        j = w_bits.index(w)
        acc[i, j] = entry["accuracy"]

    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("viridis")
    im = plt.imshow(acc, aspect="equal", cmap=cmap, interpolation="nearest")

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(w_bits) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(a_bits) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            ax.text(
                j,
                i,
                f"{acc[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=20,
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
            )

    ax.set_xticks(range(len(w_bits)))
    ax.set_yticks(range(len(a_bits)))
    ax.set_xticklabels(w_bits, fontsize=18)
    ax.set_yticklabels(a_bits, fontsize=18)
    ax.tick_params(length=0)

    plt.xlabel("Weight bits", fontsize=20, labelpad=15)
    plt.ylabel("Activation bits", fontsize=20, labelpad=15)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Accuracy (%)", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"accuracy_heatmap_{args.json_path.split('/')[-1].split('.')[0]}.pdf")
    print(
        f"Saved styled heatmap to accuracy_heatmap_{args.json_path.split('/')[-1].split('.')[0]}.pdf"
    )


if __name__ == "__main__":
    main()
