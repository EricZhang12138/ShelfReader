"""
Extract gate value distributions from the trained gated fusion model.
Usage: python extract_gate_values.py
"""

import torch
from torch.amp import autocast
from config import Config
from dataset import create_dataloaders
from models import build_model

ckpt = torch.load("checkpoints/best_model_gated_freeze4.pth", map_location="cuda", weights_only=False)
config = ckpt["config"]
config.data_root = "/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"
config.ocr_cache_path = "ocr_cache_paddle.json"

model = build_model(config).cuda()
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

_, _, test_loader = create_dataloaders(config)

gate_vals = []
has_text_list = []

for batch in test_loader:
    imgs = batch["image"].cuda()
    ids = batch["input_ids"].cuda()
    mask = batch["attention_mask"].cuda()

    # Check if sample has real text (not just [UNK])
    # [UNK] token id is 100 in BERT tokenizers; if input_ids[1] == 102 ([SEP]), it's just [CLS][SEP] padding
    has_text = (ids[:, 1] != 102)

    with torch.no_grad():
        emb = model.get_embeddings(imgs, ids, mask)
        x_img, x_txt = emb["x_img"], emb["x_txt"]
        h_img = model.fusion.img_proj(x_img)
        h_txt = model.fusion.txt_proj(x_txt)
        g = model.fusion.gate(torch.cat([h_img, h_txt], dim=-1))
        gate_vals.append(g.cpu().mean(dim=-1))  # mean gate value per sample
        has_text_list.append(has_text.cpu())

gate_vals = torch.cat(gate_vals)
has_text = torch.cat(has_text_list)

text_gates = gate_vals[has_text]
no_text_gates = gate_vals[~has_text]

print(f"Overall:  mean={gate_vals.mean():.4f}, std={gate_vals.std():.4f}")
print(f"With text ({has_text.sum()} samples):    mean={text_gates.mean():.4f}, std={text_gates.std():.4f}")
print(f"Without text ({(~has_text).sum()} samples): mean={no_text_gates.mean():.4f}, std={no_text_gates.std():.4f}")
print(f"\nNote: gate near 1.0 = rely on image, gate near 0.0 = rely on text")
