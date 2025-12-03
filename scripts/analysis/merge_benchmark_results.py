#!/usr/bin/env python3
"""
Merge Benchmark Results

After running benchmark_single_model.py for all 4 models, use this script
to merge the results and generate combined plots.

Usage:
    python merge_benchmark_results.py

Author: Project 302006
Date: December 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_results(results_dir, output_dir=None):
    """Merge all benchmark CSV files"""

    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir
    output_dir = Path(output_dir)

    # Find all individual model CSV files
    csv_files = list(results_dir.glob('benchmark_*.csv'))

    # Filter out already merged files
    csv_files = [f for f in csv_files if 'merged' not in f.name and 'summary' not in f.name]

    if not csv_files:
        logger.error(f"No benchmark CSV files found in {results_dir}")
        return None

    logger.info(f"Found {len(csv_files)} result files:")
    for f in csv_files:
        logger.info(f"  - {f.name}")

    # Load and merge all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        logger.info(f"  Loaded {len(df)} rows from {csv_file.name}")

    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (keep latest)
    merged_df = merged_df.drop_duplicates(
        subset=['Task', 'Embedding', 'Classifier'],
        keep='last'
    )

    logger.info(f"\nMerged: {len(merged_df)} unique results")
    logger.info(f"Models: {merged_df['Embedding'].unique().tolist()}")
    logger.info(f"Tasks: {merged_df['Task'].unique().tolist()}")
    logger.info(f"Classifiers: {merged_df['Classifier'].unique().tolist()}")

    # Save merged CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = output_dir / f'benchmark_merged_{timestamp}.csv'
    merged_df.to_csv(merged_file, index=False)
    logger.info(f"\nSaved merged results: {merged_file}")

    return merged_df


def plot_results(df, output_dir):
    """Generate visualization plots from merged results"""

    output_dir = Path(output_dir)

    if df.empty:
        logger.warning("No results to plot")
        return

    # 1. Comparison by embedding model (best classifier for each)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, task in enumerate(['binary', '3-class', '6-class']):
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue

        best_df = task_df.loc[task_df.groupby('Embedding')['Balanced_Accuracy'].idxmax()]

        ax = axes[idx]
        x = np.arange(len(best_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, best_df['Balanced_Accuracy'], width, label='Balanced Acc', color='steelblue')
        bars2 = ax.bar(x + width/2, best_df['AUC_Macro'].fillna(0), width, label='AUC Macro', color='coral')

        ax.set_xlabel('Embedding Model')
        ax.set_ylabel('Score')
        ax.set_title(f'{task} Classification')
        ax.set_xticks(x)
        ax.set_xticklabels(best_df['Embedding'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'embedding_comparison.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved: embedding_comparison.png")

    # 2. Heatmaps for each task
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]

        pivot_df = task_df.pivot_table(
            index='Embedding', columns='Classifier',
            values='Balanced_Accuracy', aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{task} Classification - Balanced Accuracy')
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{task}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: heatmap_{task}.png")

    # 3. Bar plot comparing all classifiers
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, task in enumerate(['binary', '3-class', '6-class']):
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue

        ax = axes[idx]
        pivot = task_df.pivot(index='Classifier', columns='Embedding', values='Balanced_Accuracy')
        pivot.plot(kind='bar', ax=ax, rot=45)

        ax.set_title(f'{task} Classification')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_ylim(0, 1)
        ax.legend(title='Embedding', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: classifier_comparison.png")


def print_summary(df, output_dir):
    """Print and save summary of best results"""

    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("COVID SEVERITY PREDICTION BENCHMARK - COMBINED RESULTS")
    summary_lines.append("=" * 80)

    # Best results per task
    summary_lines.append("\nBEST RESULTS BY TASK:")
    summary_lines.append("-" * 60)

    for task in ['binary', '3-class', '6-class']:
        task_df = df[df['Task'] == task]
        if task_df.empty:
            continue

        best_idx = task_df['Balanced_Accuracy'].idxmax()
        best = task_df.loc[best_idx]

        summary_lines.append(f"\n{task.upper()}:")
        summary_lines.append(f"  Best: {best['Embedding']} + {best['Classifier']}")
        summary_lines.append(f"  Balanced Accuracy: {best['Balanced_Accuracy']:.4f}")
        summary_lines.append(f"  AUC Macro: {best['AUC_Macro']:.4f}")
        summary_lines.append(f"  F1 Macro: {best['F1_Macro']:.4f}")

    # Embedding ranking
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("EMBEDDING RANKING (by average Balanced Accuracy):")
    summary_lines.append("-" * 60)

    ranking = df.groupby('Embedding')['Balanced_Accuracy'].mean().sort_values(ascending=False)
    for i, (emb, score) in enumerate(ranking.items(), 1):
        summary_lines.append(f"  {i}. {emb}: {score:.4f}")

    # Classifier ranking
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("CLASSIFIER RANKING (by average Balanced Accuracy):")
    summary_lines.append("-" * 60)

    clf_ranking = df.groupby('Classifier')['Balanced_Accuracy'].mean().sort_values(ascending=False)
    for i, (clf, score) in enumerate(clf_ranking.items(), 1):
        summary_lines.append(f"  {i}. {clf}: {score:.4f}")

    # Print to console
    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    # Save to file
    summary_file = output_dir / f'benchmark_summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    logger.info(f"\nSaved summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Merge COVID Severity Benchmark Results')
    parser.add_argument('--results-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006/05_RESULTS/covid_severity_benchmark",
                        help='Directory containing benchmark result CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results-dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir

    # Merge results
    df = merge_results(args.results_dir, output_dir)

    if df is not None and not df.empty:
        # Generate plots
        plot_results(df, output_dir)

        # Print summary
        print_summary(df, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
