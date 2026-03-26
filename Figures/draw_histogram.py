import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('/mnt/external/home/YuouZhang/Documents_external/shelfreader/multimodel/ShelfReader/Figures/per_category_accuracy.csv')

categories = df['category'].tolist()
models = ['Image-only', 'Concat', 'Gated', 'Cross-attn', 'Score fusion']
acc_cols = ['image_only_acc', 'concat_acc', 'gated_acc', 'cross_attention_acc', 'score_fusion_acc']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

x = np.arange(len(categories))
width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))

for i, (model, col, color) in enumerate(zip(models, acc_cols, colors)):
    vals = df[col].tolist()
    offset = (i - 2) * width
    bars = ax.bar(x + offset, vals, width, label=model, color=color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Category Test Accuracy by Fusion Architecture', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 105)
ax.legend(loc='upper left')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/external/home/YuouZhang/Documents_external/shelfreader/multimodel/ShelfReader/Figures/per_category_accuracy.png', dpi=150)
print("Saved per_category_accuracy.png")
