from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path('/Users/kareem/PycharmProjects/deltaML_Paper/trillium_project/baseline_validation')
CSV = BASE / 'baseline_val_outlier_fairness_summary.csv'
OUT_PNG = BASE / 'baseline_val_outlier_fairness_comparison.png'

order = ['BoE', 'BoB', 'BoB+BoE', 'PRE', 'PRE+BoE']
colors = {
    'BoE': '#7a8f3b',
    'BoB': '#c46c2b',
    'BoB+BoE': '#8b3f2f',
    'PRE': '#2f6c8f',
    'PRE+BoE': '#1f3f5b',
}

df = pd.read_csv(CSV)
df['Model'] = pd.Categorical(df['Model'], categories=order, ordered=True)
df = df.sort_values('Model').reset_index(drop=True)

plt.rcParams.update({
    'figure.dpi': 160,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
x = range(len(df))

# Panel 1: MAE before/after
ax = axes[0]
for i, row in df.iterrows():
    c = colors[str(row['Model'])]
    ax.plot([i, i], [row['MAE_before'], row['MAE_after']], color=c, alpha=0.35, lw=2)
    ax.scatter(i, row['MAE_before'], color='white', edgecolor=c, s=80, lw=1.8, zorder=3, label='Before' if i == 0 else None)
    ax.scatter(i, row['MAE_after'], color=c, edgecolor='black', s=80, lw=0.6, zorder=4, label='After' if i == 0 else None)
    ax.text(i, row['MAE_after'] * 1.18, f"{row['MAE_after']:.4f}", ha='center', va='bottom', fontsize=9, rotation=0)
ax.set_yscale('log')
ax.set_xticks(list(x), df['Model'])
ax.set_ylabel('Validation MAE (Ha)')
ax.set_title('Shared-mask validation MAE')
ax.grid(axis='y', alpha=0.25, which='both')
ax.legend(frameon=False, loc='upper right')

# Panel 2: R2 before/after
ax = axes[1]
for i, row in df.iterrows():
    c = colors[str(row['Model'])]
    ax.plot([i, i], [row['R2_before'], row['R2_after']], color=c, alpha=0.35, lw=2)
    ax.scatter(i, row['R2_before'], color='white', edgecolor=c, s=80, lw=1.8, zorder=3)
    ax.scatter(i, row['R2_after'], color=c, edgecolor='black', s=80, lw=0.6, zorder=4)
    ax.text(i, row['R2_after'] + 0.003, f"{row['R2_after']:.3f}", ha='center', va='bottom', fontsize=9)
ax.set_yscale('symlog', linthresh=0.01)
ax.set_xticks(list(x), df['Model'])
ax.set_ylabel('Validation $R^2$')
ax.set_title('Shared-mask validation $R^2$')
ax.grid(axis='y', alpha=0.25, which='both')
ax.axhline(0, color='black', lw=0.8, alpha=0.4)

removed = int(df['Removed_count'].iloc[0])
removed_pct = float(df['Removed_percent'].iloc[0])
fig.suptitle(f'Fair shared outlier mask across baseline validation predictions ({removed} molecules removed, {removed_pct:.3f}%)', fontsize=13)

fig.savefig(OUT_PNG, bbox_inches='tight')
print(OUT_PNG)
