"""
Member A - Data & Distributions
Runnable script version (mirrors the notebook).
Loads from the Kaggle CSV at ../data/wdbc.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

FIG_DIR = '../figures'
DATA_DIR = '../data'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

PALETTE = {'Malignant': '#D62728', 'Benign': '#1F77B4'}

# ---------- Load Kaggle CSV ----------
KAGGLE_CSV = '../data/wdbc.csv'
if not os.path.exists(KAGGLE_CSV):
    raise FileNotFoundError(
        f'Could not find {KAGGLE_CSV}.\n'
        f'Download data.csv from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data\n'
        f'rename it to wdbc.csv, and place it in the ../data/ folder.'
    )

raw = pd.read_csv(KAGGLE_CSV)
print(f'Raw Kaggle file shape: {raw.shape}')

# ---------- Clean ----------
df = raw.copy()
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])
if 'id' in df.columns:
    df = df.drop(columns=['id'])
df['diagnosis'] = df['diagnosis'].map({'M': 'Malignant', 'B': 'Benign'})
feature_cols = [c for c in df.columns if c != 'diagnosis']
df = df[feature_cols + ['diagnosis']]
df.to_csv(f'{DATA_DIR}/wdbc_clean.csv', index=False)
print(f'Cleaned: {df.shape}')

# ---------- Quality ----------
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Duplicates:     {df.duplicated().sum()}')

class_counts = df['diagnosis'].value_counts()
class_pct = df['diagnosis'].value_counts(normalize=True) * 100
print('\nClass balance:')
print(pd.DataFrame({'Count': class_counts, 'Percentage': class_pct.round(2)}))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_counts.index, class_counts.values,
              color=[PALETTE[x] for x in class_counts.index], edgecolor='black')
for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({pct:.1f}%)', ha='center', fontweight='bold')
ax.set_title('Class Balance: Malignant vs. Benign', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of samples')
ax.set_ylim(0, max(class_counts.values) * 1.15)
plt.savefig(f'{FIG_DIR}/01_class_balance.png')
plt.close()

# ---------- Descriptive stats ----------
def descriptive_stats(group_df, feature_cols):
    rows = []
    for col in feature_cols:
        s = group_df[col]
        rows.append({
            'feature': col, 'mean': s.mean(), 'median': s.median(),
            'std': s.std(), 'min': s.min(), 'max': s.max(),
            'skewness': stats.skew(s), 'kurtosis': stats.kurtosis(s),
        })
    return pd.DataFrame(rows).set_index('feature')

stats_overall = descriptive_stats(df, feature_cols)
stats_malignant = descriptive_stats(df[df['diagnosis']=='Malignant'], feature_cols)
stats_benign = descriptive_stats(df[df['diagnosis']=='Benign'], feature_cols)
stats_overall.round(4).to_csv(f'{DATA_DIR}/stats_overall.csv')
stats_malignant.round(4).to_csv(f'{DATA_DIR}/stats_malignant.csv')
stats_benign.round(4).to_csv(f'{DATA_DIR}/stats_benign.csv')

comparison = pd.DataFrame({
    'mean_malignant': stats_malignant['mean'],
    'mean_benign': stats_benign['mean'],
    'std_malignant': stats_malignant['std'],
    'std_benign': stats_benign['std'],
})
pooled_std = np.sqrt((comparison['std_malignant']**2 + comparison['std_benign']**2) / 2)
comparison['mean_diff'] = comparison['mean_malignant'] - comparison['mean_benign']
comparison['standardized_diff'] = comparison['mean_diff'] / pooled_std
comparison_sorted = comparison.reindex(
    comparison['standardized_diff'].abs().sort_values(ascending=False).index)
comparison_sorted.round(4).to_csv(f'{DATA_DIR}/group_comparison.csv')

print('\nTop 10 features by |standardized mean difference|:')
print(comparison_sorted[['mean_malignant','mean_benign','standardized_diff']].head(10).round(3))

# ---------- Distribution plots ----------
def plot_distributions(df, features, title, filename, ncols=5):
    n = len(features); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.5*nrows))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color in PALETTE.items():
            subset = df[df['diagnosis']==label][feat]
            ax.hist(subset, bins=25, alpha=0.5, color=color, label=label,
                    density=True, edgecolor='white', linewidth=0.4)
        for label, color in PALETTE.items():
            subset = df[df['diagnosis']==label][feat]
            kde = stats.gaussian_kde(subset)
            x_range = np.linspace(subset.min(), subset.max(), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=1.8)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0: ax.legend(fontsize=8)
    for j in range(n, len(axes)): axes[j].axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/{filename}')
    plt.close()

mean_features = [c for c in feature_cols if c.endswith('_mean')]
se_features = [c for c in feature_cols if c.endswith('_se')]
worst_features = [c for c in feature_cols if c.endswith('_worst')]

plot_distributions(df, mean_features, 'Distribution of MEAN features by diagnosis', '02_dist_mean_features.png')
plot_distributions(df, se_features, 'Distribution of STANDARD ERROR features by diagnosis', '03_dist_se_features.png')
plot_distributions(df, worst_features, 'Distribution of WORST features by diagnosis', '04_dist_worst_features.png')

# ---------- Boxplots ----------
def plot_boxplots(df, features, title, filename, ncols=5):
    n = len(features); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.8*nrows))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=df, x='diagnosis', y=feat, ax=ax,
                    palette=PALETTE, hue='diagnosis',
                    order=['Benign', 'Malignant'], legend=False,
                    width=0.55, fliersize=3)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(labelsize=8)
    for j in range(n, len(axes)): axes[j].axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/{filename}')
    plt.close()

plot_boxplots(df, mean_features, 'Boxplots — MEAN features', '05_box_mean_features.png')
plot_boxplots(df, se_features, 'Boxplots — STANDARD ERROR features', '06_box_se_features.png')
plot_boxplots(df, worst_features, 'Boxplots — WORST features', '07_box_worst_features.png')

# ---------- Top 6 spotlight ----------
top_features = comparison_sorted.index[:6].tolist()
print(f'\nTop 6 features:\n  ' + '\n  '.join(top_features))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    ax = axes[i]
    for label, color in PALETTE.items():
        subset = df[df['diagnosis']==label][feat]
        ax.hist(subset, bins=30, alpha=0.55, color=color, label=label,
                density=True, edgecolor='white')
        kde = stats.gaussian_kde(subset)
        x_range = np.linspace(subset.min(), subset.max(), 200)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2.2)
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density')
    if i == 0: ax.legend()
fig.suptitle('Top 6 most discriminative features — distributions',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_top6_distributions.png')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    ax = axes[i]
    sns.boxplot(data=df, x='diagnosis', y=feat, ax=ax,
                palette=PALETTE, hue='diagnosis',
                order=['Benign', 'Malignant'], legend=False, width=0.5)
    sns.stripplot(data=df, x='diagnosis', y=feat, ax=ax,
                  order=['Benign', 'Malignant'],
                  color='black', size=2.0, alpha=0.4, jitter=0.2)
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
fig.suptitle('Top 6 most discriminative features — boxplots',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/09_top6_boxplots.png')
plt.close()

print('\nAll outputs written to ../data and ../figures.')
print('Done.')
