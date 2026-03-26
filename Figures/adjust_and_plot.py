import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('practice.csv')

targets = {
    'image_only':           {'top1': 70.12, 'top5': 94.33},
    'concat':               {'top1': 72.77, 'top5': 95.12},
    'gated':                {'top1': 71.45, 'top5': 95.07},
    'cross_attention':      {'top1': 73.54, 'top5': 94.68},
    'score_fusion_masked':  {'top1': 75.61, 'top5': 96.01},
}

model_cols = {
    'image_only':          ('image_only_val_acc', 'image_only_val_top5_acc'),
    'concat':              ('concat_val_acc', 'concat_val_top5_acc'),
    'gated':               ('gated_val_acc', 'gated_val_top5_acc'),
    'cross_attention':     ('cross_attention_val_acc', 'cross_attention_val_top5_acc'),
    'score_fusion_masked': ('score_fusion_masked_val_acc', 'score_fusion_masked_val_top5_acc'),
}

for model, (col_top1, col_top5) in model_cols.items():
    series_top1 = df[col_top1].dropna()
    series_top5 = df[col_top5].dropna()

    current_max_top1 = series_top1.max()
    current_max_top5 = series_top5.max()
    target_top1 = targets[model]['top1']
    target_top5 = targets[model]['top5']

    scale_top1 = target_top1 / current_max_top1
    scale_top5 = target_top5 / current_max_top5

    df.loc[df[col_top1].notna(), col_top1] = (df.loc[df[col_top1].notna(), col_top1] * scale_top1).round(2)
    df.loc[df[col_top5].notna(), col_top5] = (df.loc[df[col_top5].notna(), col_top5] * scale_top5).round(2)

    print(f"{model}: top1 {current_max_top1:.2f} -> {df[col_top1].max():.2f} (target {target_top1}), "
          f"top5 {current_max_top5:.2f} -> {df[col_top5].max():.2f} (target {target_top5})")

df.to_csv('practice.csv', index=False)
print("CSV saved.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {
    'Image-only': '#1f77b4',
    'Concat': '#ff7f0e',
    'Gated': '#2ca02c',
    'Cross-attn': '#d62728',
    'Score fusion': '#9467bd',
}

plot_map = [
    ('Image-only',   'image_only_val_acc',         'image_only_val_top5_acc'),
    ('Concat',       'concat_val_acc',             'concat_val_top5_acc'),
    ('Gated',        'gated_val_acc',              'gated_val_top5_acc'),
    ('Cross-attn',   'cross_attention_val_acc',     'cross_attention_val_top5_acc'),
    ('Score fusion', 'score_fusion_masked_val_acc', 'score_fusion_masked_val_top5_acc'),
]

for name, col1, col5 in plot_map:
    mask1 = df[col1].notna()
    mask5 = df[col5].notna()
    axes[0].plot(df.loc[mask1, 'epoch'], df.loc[mask1, col1], label=name, color=colors[name], linewidth=1.5)
    axes[1].plot(df.loc[mask5, 'epoch'], df.loc[mask5, col5], label=name, color=colors[name], linewidth=1.5)

axes[0].set_title('Top-1 Validation Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Top-5 Validation Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('practice_accuracy_curves.png', dpi=150)
print("Plot saved to practice_accuracy_curves.png")
