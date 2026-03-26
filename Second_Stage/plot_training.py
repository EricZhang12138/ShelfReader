"""
Generate paper-ready figures from saved JSON training histories and checkpoints.

Produces (saved to figures/):
  1. Training curves — loss + accuracy over epochs for all models (combined)
  2. OCR coverage by category — bar chart (Fruit, Packages, Vegetables)
  3. Per-category accuracy comparison — image-only vs best fusion per category
  4. Score fusion alpha over training — text_scale growth over epochs
  5. (Gate distribution already exists as gate_distribution.png)

Usage:
    # All plots except per-category accuracy (no GPU needed):
    python plot_training.py --log_dir logs --ocr_cache ocr_cache_paddlev5.json \
        --data_root /path/to/GroceryStoreDataset/dataset

    # Include per-category accuracy (needs GPU + checkpoints):
    python plot_training.py --log_dir logs --per_category \
        --data_root /path/to/GroceryStoreDataset/dataset \
        --ocr_cache ocr_cache_paddlev5.json
"""

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────────

STYLE = {
    "image_only":            {"color": "#7f7f7f", "label": "Image-Only",       "ls": "-",  "marker": "o"},
    "concat":                {"color": "#1f77b4", "label": "Concat",           "ls": "-",  "marker": "s"},
    "gated":                 {"color": "#2ca02c", "label": "Gated",            "ls": "-",  "marker": "^"},
    "cross_attention":       {"color": "#d62728", "label": "Cross-Attention",  "ls": "-",  "marker": "D"},
    "score_fusion_masked":   {"color": "#9467bd", "label": "Score Fusion",     "ls": "-",  "marker": "v"},
}

CATEGORY_COLORS = {
    "Fruit":      "#2ca02c",
    "Packages":   "#1f77b4",
    "Vegetables": "#ff7f0e",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.figsize": (7, 5),
})


# ── Helpers ──────────────────────────────────────────────────────────────


def load_histories(log_dir):
    """Load all history_*.json files and return {arch_name: [records]}."""
    histories = {}
    for path in sorted(glob(os.path.join(log_dir, "history_*.json"))):
        with open(path) as f:
            data = json.load(f)
        name = os.path.basename(path).replace("history_", "").replace(".json", "")
        if name.startswith("score_fusion"):
            pass  # keep as-is, e.g. "score_fusion_masked"
        elif "_freeze" in name:
            name = name.split("_freeze")[0]
        histories[name] = data
    return histories


def get_style(name):
    return STYLE.get(name, {"color": "black", "label": name, "ls": "--", "marker": "x"})


def build_label_to_category(data_root):
    """Map fine_label (int) -> coarse category (Fruit/Packages/Vegetables) from train.txt."""
    mapping = {}
    split_file = os.path.join(data_root, "train.txt")
    with open(split_file) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            fine_label = int(parts[1])
            coarse = parts[0].split("/")[1]  # e.g. "train/Fruit/..." -> "Fruit"
            mapping[fine_label] = coarse
    return mapping


# ── Figure 1: Training Curves (loss + accuracy) ─────────────────────────


def plot_training_curves(histories, out_dir):
    """Combined 2-panel figure: val loss (left) + val accuracy (right) for all archs."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    for name, records in histories.items():
        style = get_style(name)
        epochs = [r["epoch"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        val_acc = [r["val_acc"] for r in records]
        me = max(1, len(epochs) // 10)

        ax_loss.plot(epochs, val_loss, label=style["label"], color=style["color"],
                     ls=style["ls"], marker=style["marker"], markevery=me, markersize=5)
        ax_acc.plot(epochs, val_acc, label=style["label"], color=style["color"],
                    ls=style["ls"], marker=style["marker"], markevery=me, markersize=5)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.set_title("(a) Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy (%)")
    ax_acc.set_title("(b) Validation Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"))
    fig.savefig(os.path.join(out_dir, "training_curves.pdf"))
    plt.close(fig)
    print("  Saved training_curves.png/pdf")


def plot_training_curves_with_train(histories, out_dir):
    """4-panel figure: train loss, val loss, train acc, val acc."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_tl, ax_vl = axes[0]
    ax_ta, ax_va = axes[1]

    for name, records in histories.items():
        style = get_style(name)
        epochs = [r["epoch"] for r in records]
        me = max(1, len(epochs) // 10)

        ax_tl.plot(epochs, [r["train_loss"] for r in records],
                   label=style["label"], color=style["color"],
                   marker=style["marker"], markevery=me, markersize=4)
        ax_vl.plot(epochs, [r["val_loss"] for r in records],
                   label=style["label"], color=style["color"],
                   marker=style["marker"], markevery=me, markersize=4)
        ax_ta.plot(epochs, [r["train_acc"] for r in records],
                   label=style["label"], color=style["color"],
                   marker=style["marker"], markevery=me, markersize=4)
        ax_va.plot(epochs, [r["val_acc"] for r in records],
                   label=style["label"], color=style["color"],
                   marker=style["marker"], markevery=me, markersize=4)

    for ax, title, ylabel in [
        (ax_tl, "(a) Training Loss", "Loss"),
        (ax_vl, "(b) Validation Loss", "Loss"),
        (ax_ta, "(c) Training Accuracy", "Accuracy (%)"),
        (ax_va, "(d) Validation Accuracy", "Accuracy (%)"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves_full.png"))
    fig.savefig(os.path.join(out_dir, "training_curves_full.pdf"))
    plt.close(fig)
    print("  Saved training_curves_full.png/pdf")


# ── Figure 3: OCR Coverage by Category ──────────────────────────────────


def plot_ocr_coverage(data_root, ocr_cache_path, out_dir):
    """Bar chart of OCR text coverage per coarse category."""
    with open(ocr_cache_path) as f:
        ocr_cache = json.load(f)

    cats = {"Fruit": [0, 0], "Packages": [0, 0], "Vegetables": [0, 0]}

    # Count across all splits
    for split in ["train", "val", "test"]:
        split_file = os.path.join(data_root, f"{split}.txt")
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                rel_path = parts[0]
                coarse = rel_path.split("/")[1]
                if coarse not in cats:
                    continue
                cats[coarse][1] += 1
                ocr_text = ocr_cache.get(rel_path, "")
                if ocr_text.strip():
                    cats[coarse][0] += 1

    categories = list(cats.keys())
    has_text = [cats[c][0] for c in categories]
    totals = [cats[c][1] for c in categories]
    pcts = [100.0 * h / t if t > 0 else 0 for h, t in zip(has_text, totals)]
    colors = [CATEGORY_COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, pcts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, pct, h, t in zip(bars, pcts, has_text, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{pct:.0f}%\n({h}/{t})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Samples with OCR Text (%)")
    ax.set_title("OCR Text Coverage by Product Category")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "ocr_coverage_by_category.png"))
    fig.savefig(os.path.join(out_dir, "ocr_coverage_by_category.pdf"))
    plt.close(fig)
    print("  Saved ocr_coverage_by_category.png/pdf")


# ── Figure 4: Per-Category Accuracy Comparison ──────────────────────────


def plot_per_category_accuracy(args, out_dir):
    """Bar chart comparing image-only vs best fusion model per category."""
    import torch
    from torch.amp import autocast
    from tqdm import tqdm

    from config import Config
    from dataset import create_dataloaders
    from models import build_model
    from train_image_only import ImageOnlyClassifier

    label_to_cat = build_label_to_category(args.data_root)

    def eval_per_category(model_fn, test_loader, config):
        """Run model and return {category: (correct, total)}."""
        cat_stats = {"Fruit": [0, 0], "Packages": [0, 0], "Vegetables": [0, 0]}
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                images = batch["image"].to(config.device)
                labels = batch["label"]

                with autocast("cuda", enabled=config.fp16):
                    logits = model_fn(batch, config)

                preds = logits.argmax(1).cpu()
                for pred, label in zip(preds, labels):
                    cat = label_to_cat.get(label.item(), None)
                    if cat and cat in cat_stats:
                        cat_stats[cat][1] += 1
                        if pred.item() == label.item():
                            cat_stats[cat][0] += 1
        return {c: 100.0 * s[0] / s[1] if s[1] > 0 else 0 for c, s in cat_stats.items()}

    # Load image-only model
    img_ckpt_path = os.path.join(args.ckpt_dir, "best_model_image_only.pth")
    img_checkpoint = torch.load(img_ckpt_path, map_location=args.device, weights_only=False)
    img_config = img_checkpoint.get("config", Config())
    img_config.device = args.device
    img_config.data_root = args.data_root
    img_config.ocr_cache_path = args.ocr_cache

    img_model = ImageOnlyClassifier(img_config).to(img_config.device)
    img_model.load_state_dict(img_checkpoint["model_state_dict"], strict=False)
    img_model.eval()

    _, _, test_loader = create_dataloaders(img_config)

    print("  Evaluating image-only per category...")
    img_accs = eval_per_category(
        lambda batch, cfg: img_model(batch["image"].to(cfg.device)),
        test_loader, img_config
    )

    # Find best fusion model (highest val_acc from checkpoints)
    best_fusion_name = None
    best_fusion_acc = 0
    for path in glob(os.path.join(args.ckpt_dir, "best_model_*.pth")):
        basename = os.path.basename(path)
        if "image_only" in basename or "score_fusion" in basename:
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        acc = ckpt.get("val_acc", 0)
        if acc > best_fusion_acc:
            best_fusion_acc = acc
            best_fusion_name = basename
            best_fusion_path = path

    if best_fusion_name is None:
        print("  No fusion checkpoints found, skipping per-category plot.")
        return

    # Derive display name
    name_key = best_fusion_name.replace("best_model_", "").replace(".pth", "")
    if "_freeze" in name_key:
        name_key = name_key.split("_freeze")[0]
    fusion_style = get_style(name_key)

    fusion_checkpoint = torch.load(best_fusion_path, map_location=args.device, weights_only=False)
    fusion_config = fusion_checkpoint.get("config", Config())
    fusion_config.device = args.device
    fusion_config.data_root = args.data_root
    fusion_config.ocr_cache_path = args.ocr_cache

    fusion_model = build_model(fusion_config).to(fusion_config.device)
    fusion_model.load_state_dict(fusion_checkpoint["model_state_dict"], strict=False)
    fusion_model.eval()

    _, _, test_loader2 = create_dataloaders(fusion_config)

    print(f"  Evaluating {fusion_style['label']} per category...")
    fusion_accs = eval_per_category(
        lambda batch, cfg: fusion_model(
            batch["image"].to(cfg.device),
            batch["input_ids"].to(cfg.device),
            batch["attention_mask"].to(cfg.device),
        ),
        test_loader2, fusion_config
    )

    # Plot grouped bar chart
    categories = ["Fruit", "Packages", "Vegetables"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, [img_accs[c] for c in categories], width,
                   label="Image-Only", color="#7f7f7f", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [fusion_accs[c] for c in categories], width,
                   label=fusion_style["label"], color=fusion_style["color"],
                   edgecolor="black", linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add delta labels
    for i, cat in enumerate(categories):
        delta = fusion_accs[cat] - img_accs[cat]
        sign = "+" if delta >= 0 else ""
        y_pos = max(img_accs[cat], fusion_accs[cat]) + 4
        ax.text(x[i], y_pos, f"{sign}{delta:.1f}%",
                ha="center", va="bottom", fontsize=9, color="red" if delta < 0 else "green",
                fontweight="bold")

    ax.set_xlabel("Product Category")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Category Accuracy: Image-Only vs Best Fusion")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(img_accs.values()), max(fusion_accs.values())) + 10)

    fig.savefig(os.path.join(out_dir, "per_category_accuracy.png"))
    fig.savefig(os.path.join(out_dir, "per_category_accuracy.pdf"))
    plt.close(fig)
    print("  Saved per_category_accuracy.png/pdf")


# ── Figure 5: Score Fusion Alpha Over Training ──────────────────────────


def plot_score_fusion_alpha(histories, out_dir):
    """Plot learned text_scale (alpha) over epochs for score fusion."""
    # Find score fusion history with text_scale
    sf_records = None
    for name, records in histories.items():
        if "score_fusion" in name and records and "text_scale" in records[0]:
            sf_records = records
            break

    if sf_records is None:
        print("  No score fusion history with text_scale found, skipping alpha plot.")
        return

    epochs = [r["epoch"] for r in sf_records]
    alphas = [r["text_scale"] for r in sf_records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, alphas, color="#9467bd", linewidth=2, marker="v",
            markevery=max(1, len(epochs) // 10), markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned Text Scale ($\\alpha$)")
    ax.set_title("Score Fusion: Learned Text Scale Over Training")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Annotate start and end
    ax.annotate(f"{alphas[0]:.4f}", (epochs[0], alphas[0]),
                textcoords="offset points", xytext=(10, -10), fontsize=9)
    ax.annotate(f"{alphas[-1]:.4f}", (epochs[-1], alphas[-1]),
                textcoords="offset points", xytext=(-40, 10), fontsize=9,
                fontweight="bold")

    fig.savefig(os.path.join(out_dir, "score_fusion_alpha.png"))
    fig.savefig(os.path.join(out_dir, "score_fusion_alpha.pdf"))
    plt.close(fig)
    print("  Saved score_fusion_alpha.png/pdf")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready training plots")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory containing history_*.json files")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to GroceryStoreDataset/dataset (needed for OCR coverage + per-category)")
    parser.add_argument("--ocr_cache", type=str, default="ocr_cache_paddlev5.json")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--per_category", action="store_true",
                        help="Generate per-category accuracy comparison (needs GPU + checkpoints)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load histories
    histories = load_histories(args.log_dir)
    if not histories:
        print(f"No history files found in {args.log_dir}/")
        return
    print(f"Found histories: {list(histories.keys())}")

    # Figure 1: Training curves
    plot_training_curves(histories, args.out_dir)
    plot_training_curves_with_train(histories, args.out_dir)

    # Figure 3: OCR coverage by category
    if args.data_root and os.path.exists(args.ocr_cache):
        plot_ocr_coverage(args.data_root, args.ocr_cache, args.out_dir)
    else:
        print("  Skipping OCR coverage (need --data_root and --ocr_cache)")

    # Figure 5: Score fusion alpha over training
    plot_score_fusion_alpha(histories, args.out_dir)

    # Figure 4: Per-category accuracy comparison (optional, needs GPU)
    if args.per_category:
        if not args.data_root:
            print("ERROR: --data_root required for per-category accuracy")
        else:
            plot_per_category_accuracy(args, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
