#!/usr/bin/env python3
"""
Create Publication-Quality Figures for COVID Severity Benchmark

Generates figures suitable for GitHub README and publications.

Usage:
    python scripts/analysis/create_benchmark_figures.py

Author: Project 302006
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_results(results_dir):
    """Load merged benchmark results"""
    results_dir = Path(results_dir)

    # Find the most recent merged CSV
    merged_files = list(results_dir.glob('benchmark_merged_*.csv'))
    if not merged_files:
        # Try loading individual files
        csv_files = list(results_dir.glob('benchmark_*.csv'))
        csv_files = [f for f in csv_files if 'merged' not in f.name and 'summary' not in f.name]
        if csv_files:
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=['Task', 'Embedding', 'Classifier'], keep='last')
            return df
        raise FileNotFoundError(f"No benchmark CSV files found in {results_dir}")

    latest = max(merged_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {latest}")
    return pd.read_csv(latest)


def fig1_model_comparison_bar(df, output_dir):
    """
    Figure 1: Model Comparison Bar Chart
    Shows Balanced Accuracy and AUC for best classifier per model
    """
    # Get best classifier for each model
    best_df = df.loc[df.groupby('Embedding')['Balanced_Accuracy'].idxmax()].copy()
    best_df = best_df.sort_values('Balanced_Accuracy', ascending=True)

    # Define colors
    colors = {
        'UCE': '#2ecc71',           # Green
        'TranscriptFormer': '#3498db',  # Blue
        'Geneformer': '#9b59b6',    # Purple
        'scGPT': '#e74c3c'          # Red
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(best_df))
    height = 0.35

    # Bars
    bars1 = ax.barh(y - height/2, best_df['Balanced_Accuracy'], height,
                    label='Balanced Accuracy', color=[colors[e] for e in best_df['Embedding']],
                    alpha=0.9, edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y + height/2, best_df['AUC_Macro'], height,
                    label='AUC', color=[colors[e] for e in best_df['Embedding']],
                    alpha=0.5, edgecolor='black', linewidth=0.5, hatch='///')

    # Labels
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_ylabel('Foundation Model', fontweight='bold')
    ax.set_title('COVID-19 Severity Prediction: Foundation Model Comparison\n(Binary Classification: Severe vs Non-Severe)',
                 fontweight='bold', pad=20)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['Embedding']}\n({row['Classifier']})" for _, row in best_df.iterrows()])
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right')

    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')

    for bar in bars2:
        width = bar.get_width()
        ax.annotate(f'{width:.3f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10)

    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'fig1_model_comparison.png')
    plt.savefig(output_dir / 'fig1_model_comparison.pdf')
    plt.close()
    print(f"Saved: fig1_model_comparison.png/pdf")


def fig2_heatmap(df, output_dir):
    """
    Figure 2: Heatmap of all model-classifier combinations
    """
    # Pivot for heatmap
    pivot_df = df.pivot_table(
        index='Embedding', columns='Classifier',
        values='Balanced_Accuracy', aggfunc='mean'
    )

    # Reorder
    model_order = ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']
    pivot_df = pivot_df.reindex([m for m in model_order if m in pivot_df.index])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=0.85, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Balanced Accuracy'})

    ax.set_title('COVID-19 Severity Prediction Performance\n(Balanced Accuracy)',
                 fontweight='bold', pad=20)
    ax.set_xlabel('Classifier', fontweight='bold')
    ax.set_ylabel('Foundation Model', fontweight='bold')

    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'fig2_heatmap.png')
    plt.savefig(output_dir / 'fig2_heatmap.pdf')
    plt.close()
    print(f"Saved: fig2_heatmap.png/pdf")


def fig3_auc_comparison(df, output_dir):
    """
    Figure 3: AUC Comparison with error-bar style visualization
    """
    # Get best result per model
    best_df = df.loc[df.groupby('Embedding')['AUC_Macro'].idxmax()].copy()
    best_df = best_df.sort_values('AUC_Macro', ascending=False)

    # Colors
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(best_df))
    bars = ax.bar(x, best_df['AUC_Macro'], color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add horizontal line at 0.5 (random)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')

    # Labels
    ax.set_xlabel('Foundation Model', fontweight='bold')
    ax.set_ylabel('AUC (Area Under ROC Curve)', fontweight='bold')
    ax.set_title('COVID-19 Severity Prediction: AUC Comparison\n(Binary Classification)',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(best_df['Embedding'], fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='upper right')

    # Add value labels on bars
    for bar, (_, row) in zip(bars, best_df.iterrows()):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}\n({row["Classifier"][:3]})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'fig3_auc_comparison.png')
    plt.savefig(output_dir / 'fig3_auc_comparison.pdf')
    plt.close()
    print(f"Saved: fig3_auc_comparison.png/pdf")


def fig4_classifier_effect(df, output_dir):
    """
    Figure 4: Grouped bar chart showing classifier effect per model
    """
    # Pivot data
    pivot = df.pivot(index='Embedding', columns='Classifier', values='Balanced_Accuracy')

    # Reorder
    model_order = ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pivot))
    width = 0.35

    colors = ['#3498db', '#2ecc71']

    for i, clf in enumerate(pivot.columns):
        bars = ax.bar(x + i*width - width/2, pivot[clf], width,
                      label=clf, color=colors[i], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Foundation Model', fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontweight='bold')
    ax.set_title('Classifier Performance Across Foundation Models\n(Binary Classification)',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(title='Classifier', loc='upper right')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'fig4_classifier_effect.png')
    plt.savefig(output_dir / 'fig4_classifier_effect.pdf')
    plt.close()
    print(f"Saved: fig4_classifier_effect.png/pdf")


def fig5_summary_table(df, output_dir):
    """
    Figure 5: Summary table as image (for README)
    """
    # Get best per model
    best_df = df.loc[df.groupby('Embedding')['Balanced_Accuracy'].idxmax()].copy()
    best_df = best_df.sort_values('Balanced_Accuracy', ascending=False)

    # Select columns
    table_df = best_df[['Embedding', 'Classifier', 'Balanced_Accuracy', 'AUC_Macro', 'Accuracy', 'N_Samples']].copy()
    table_df.columns = ['Model', 'Classifier', 'Balanced Acc', 'AUC', 'Accuracy', 'Samples']
    table_df = table_df.round(3)
    table_df['Rank'] = range(1, len(table_df) + 1)
    table_df = table_df[['Rank', 'Model', 'Classifier', 'Balanced Acc', 'AUC', 'Accuracy', 'Samples']]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db']*len(table_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(table_df.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best row
    for i in range(len(table_df.columns)):
        table[(1, i)].set_facecolor('#d5f5e3')

    ax.set_title('COVID-19 Severity Prediction: Model Ranking\n(Binary Classification: Severe vs Non-Severe)',
                 fontweight='bold', pad=20, fontsize=14)

    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'fig5_summary_table.png')
    plt.savefig(output_dir / 'fig5_summary_table.pdf')
    plt.close()
    print(f"Saved: fig5_summary_table.png/pdf")


def create_readme_badge(df, output_dir):
    """Create a simple badge-style summary image"""
    best = df.loc[df['Balanced_Accuracy'].idxmax()]

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.axis('off')

    text = f"Best Model: {best['Embedding']} | AUC: {best['AUC_Macro']:.3f} | Balanced Acc: {best['Balanced_Accuracy']:.3f}"

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.8, edgecolor='black'))

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'badge_best_model.png', transparent=True)
    plt.close()
    print(f"Saved: badge_best_model.png")


def main():
    # Paths
    results_dir = Path("/bigdata/godziklab/shared/Xinru/302006/05_RESULTS/covid_severity_benchmark")
    output_dir = Path("/bigdata/godziklab/shared/Xinru/302006/05_RESULTS/covid_severity_benchmark/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading benchmark results...")
    df = load_results(results_dir)
    print(f"Loaded {len(df)} results")
    print(f"Models: {df['Embedding'].unique().tolist()}")
    print(f"Tasks: {df['Task'].unique().tolist()}")

    # Filter to binary task only (for now)
    df_binary = df[df['Task'] == 'binary'].copy()

    if df_binary.empty:
        print("No binary task results found!")
        return 1

    print(f"\nGenerating figures for binary classification...")

    # Generate all figures
    fig1_model_comparison_bar(df_binary, output_dir)
    fig2_heatmap(df_binary, output_dir)
    fig3_auc_comparison(df_binary, output_dir)
    fig4_classifier_effect(df_binary, output_dir)
    fig5_summary_table(df_binary, output_dir)
    create_readme_badge(df_binary, output_dir)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")

    # Copy key figures to main results directory for easy access
    import shutil
    for fig in ['fig1_model_comparison.png', 'fig3_auc_comparison.png', 'fig5_summary_table.png']:
        src = output_dir / fig
        dst = results_dir / fig
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied to: {dst}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
