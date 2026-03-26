"""
Extract and plot gate value distributions from the trained Gated Fusion model.
Splits analysis by samples with OCR text vs without.
"""

import json
import sys
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.amp import autocast

sys.path.insert(0, ".")
from config import Config
from dataset import create_dataloaders
from models import build_model


@torch.no_grad()
def extract_gate_values(model, dataloader, config, ocr_cache):
    """Run inference and capture gate values from the gated fusion module."""
    model.eval()
    gate_values_with_text = []
    gate_values_no_text = []

    dataset = dataloader.dataset

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        # Get projected features (same as fusion forward pass)
        with autocast("cuda", enabled=config.fp16):
            x_img = model.image_encoder(images)
            x_txt = model.text_encoder(input_ids, attention_mask)

            img_proj = model.fusion.img_proj(x_img)
            txt_proj = model.fusion.txt_proj(x_txt)
            combined = torch.cat([img_proj, txt_proj], dim=-1)
            gate = model.fusion.gate(combined)  # [B, fused_dim], values in [0, 1]

        gate_np = gate.cpu().float().numpy()

        # Determine which samples have OCR text
        start_idx = batch_idx * dataloader.batch_size
        for i in range(gate_np.shape[0]):
            sample_idx = start_idx + i
            if sample_idx >= len(dataset.samples):
                break
            rel_path, _ = dataset.samples[sample_idx]
            has_text = bool(ocr_cache.get(rel_path, "").strip())

            if has_text:
                gate_values_with_text.append(gate_np[i])
            else:
                gate_values_no_text.append(gate_np[i])

    return np.array(gate_values_with_text), np.array(gate_values_no_text)


def plot_gate_distribution(gate_with_text, gate_no_text, save_path="gate_distribution.png"):
    """Plot gate value distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Overall distribution (mean gate per sample)
    ax = axes[0]
    mean_with = gate_with_text.mean(axis=1)
    mean_no = gate_no_text.mean(axis=1)
    ax.hist(mean_with, bins=50, alpha=0.6, label=f"With OCR (n={len(mean_with)})", color="steelblue")
    ax.hist(mean_no, bins=50, alpha=0.6, label=f"No OCR (n={len(mean_no)})", color="coral")
    ax.set_xlabel("Mean Gate Value (per sample)")
    ax.set_ylabel("Count")
    ax.set_title("Gate Distribution: Image vs Text Weight")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.7, ax.get_ylim()[1] * 0.9, "← more text    more image →", fontsize=8, ha="center")
    ax.legend()

    # 2. Per-dimension gate values (averaged across all samples)
    ax = axes[1]
    dim_mean_with = gate_with_text.mean(axis=0)
    dim_mean_no = gate_no_text.mean(axis=0)
    dims = np.arange(len(dim_mean_with))
    ax.plot(dims, np.sort(dim_mean_with), label="With OCR", color="steelblue", alpha=0.8)
    ax.plot(dims, np.sort(dim_mean_no), label="No OCR", color="coral", alpha=0.8)
    ax.set_xlabel("Dimension (sorted)")
    ax.set_ylabel("Mean Gate Value")
    ax.set_title("Per-Dimension Gate Values")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Histogram of ALL gate values (flattened)
    ax = axes[2]
    ax.hist(gate_with_text.flatten(), bins=50, alpha=0.6, label="With OCR", color="steelblue", density=True)
    ax.hist(gate_no_text.flatten(), bins=50, alpha=0.6, label="No OCR", color="coral", density=True)
    ax.set_xlabel("Gate Value")
    ax.set_ylabel("Density")
    ax.set_title("All Gate Values (flattened)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")

    # Print summary stats
    print(f"\nGate Value Summary (1.0 = all image, 0.0 = all text):")
    print(f"  With OCR text:  mean={mean_with.mean():.4f}, std={mean_with.std():.4f}, "
          f"min={mean_with.min():.4f}, max={mean_with.max():.4f}")
    print(f"  No OCR text:    mean={mean_no.mean():.4f}, std={mean_no.std():.4f}, "
          f"min={mean_no.min():.4f}, max={mean_no.max():.4f}")
    print(f"  Difference:     {mean_no.mean() - mean_with.mean():+.4f} "
          f"(no-text samples lean {'more image' if mean_no.mean() > mean_with.mean() else 'more text'})")


def main():
    ckpt_path = "checkpoints/best_model_gated_freeze2.pth"
    ocr_cache_path = "ocr_cache_combined.json"
    data_root = "/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"

    checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    config = checkpoint.get("config", Config())
    config.device = "cuda"
    config.data_root = data_root
    config.ocr_cache_path = ocr_cache_path

    with open(ocr_cache_path) as f:
        ocr_cache = json.load(f)

    _, _, test_loader = create_dataloaders(config)

    model = build_model(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print("Extracting gate values from test set...")
    gate_with, gate_no = extract_gate_values(model, test_loader, config, ocr_cache)
    print(f"  Samples with text: {len(gate_with)}")
    print(f"  Samples without text: {len(gate_no)}")

    plot_gate_distribution(gate_with, gate_no)


if __name__ == "__main__":
    main()
