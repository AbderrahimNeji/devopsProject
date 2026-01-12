import csv
import argparse
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_results_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields when possible
            rr = {}
            for k, v in r.items():
                try:
                    rr[k] = float(v)
                except (ValueError, TypeError):
                    rr[k] = v
            rows.append(rr)
    return rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_losses(rows, out_dir: Path):
    epochs = [r["epoch"] for r in rows]
    keys = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, (ktr, kv, title) in zip(axes, keys):
        ax.plot(epochs, [r.get(ktr, None) for r in rows], label="train")
        ax.plot(epochs, [r.get(kv, None) for r in rows], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_losses.png", dpi=150)
    plt.close(fig)


def plot_pr(rows, out_dir: Path):
    epochs = [r["epoch"] for r in rows]
    precision = [r.get("metrics/precision(B)", None) for r in rows]
    recall = [r.get("metrics/recall(B)", None) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, precision, label="Precision")
    ax.plot(epochs, recall, label="Recall")
    ax.set_title("Precision & Recall vs Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_vs_epoch.png", dpi=150)
    plt.close(fig)


def plot_map(rows, out_dir: Path):
    epochs = [r["epoch"] for r in rows]
    map50 = [r.get("metrics/mAP50(B)", None) for r in rows]
    map5095 = [r.get("metrics/mAP50-95(B)", None) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, map50, label="mAP@0.50")
    ax.plot(epochs, map5095, label="mAP@0.50:0.95")
    ax.set_title("mAP vs Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Annotate best mAP50 epoch
    if map50 and any(m is not None for m in map50):
        best_idx = max(range(len(map50)), key=lambda i: map50[i])
        ax.scatter([epochs[best_idx]], [map50[best_idx]], color="red")
        ax.annotate(
            f"best@{int(epochs[best_idx])}: {map50[best_idx]:.3f}",
            (epochs[best_idx], map50[best_idx]),
            textcoords="offset points",
            xytext=(10, -10),
            ha="left",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "map_vs_epoch.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot YOLO training curves from results.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--out", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv introuvable: {csv_path}")

    # Default output under project results/training_curves
    if args.out:
        out_dir = Path(args.out)
    else:
        proj_root = Path(__file__).resolve().parents[1]
        out_dir = proj_root / "results" / "training_curves"

    ensure_dir(out_dir)

    rows = read_results_csv(csv_path)
    if not rows:
        raise RuntimeError("results.csv semble vide")

    plot_losses(rows, out_dir)
    plot_pr(rows, out_dir)
    plot_map(rows, out_dir)

    print(f"Courbes générées dans: {out_dir}")


if __name__ == "__main__":
    main()
