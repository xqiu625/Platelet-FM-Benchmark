#!/usr/bin/env python3
"""
Create Embedding Visualizations for COVID Severity

Generates UMAP, t-SNE, PCA, MDS visualizations for all foundation models.
Also creates fancy publication-quality figures.

Usage:
    python scripts/analysis/create_embedding_visualizations.py

Author: Project 302006
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import umap

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# Color palettes
SEVERITY_COLORS = {
    'healthy': '#2ecc71',      # Green
    'mild': '#3498db',         # Blue
    'moderate': '#f39c12',     # Orange
    'severe': '#e74c3c',       # Red
    'fatal': '#8e44ad',        # Purple
    'recovered': '#1abc9c',    # Teal
    'unknown': '#95a5a6'       # Gray
}

SEVERITY_ORDER = ['healthy', 'mild', 'moderate', 'severe', 'fatal', 'recovered']

BINARY_COLORS = {
    'non_severe': '#3498db',   # Blue
    'severe': '#e74c3c'        # Red
}

MODEL_COLORS = {
    'UCE': '#2ecc71',
    'scGPT': '#e74c3c',
    'TranscriptFormer': '#3498db',
    'Geneformer': '#9b59b6'
}


def load_embeddings(base_dir):
    """Load all embedding files"""
    embedding_dir = Path(base_dir) / "02_EMBEDDINGS"

    configs = {
        'UCE': {
            'files': [embedding_dir / "uce/human_platelet_covid_severity_uce_adata.h5ad"],
            'obsm_keys': ['X_uce', 'X_uce_4layer']
        },
        'scGPT': {
            'files': [embedding_dir / "scgpt/platelet_scgpt_combined.h5ad"],
            'obsm_keys': ['X_scGPT', 'X_scgpt']
        },
        'TranscriptFormer': {
            'files': [embedding_dir / "transcriptformer/covid_transcriptformer.h5ad"],
            'obsm_keys': ['X_transcriptformer', 'X_TranscriptFormer']
        },
        'Geneformer': {
            'files': [embedding_dir / "geneformer/covid_geneformer.h5ad"],
            'obsm_keys': ['X_geneformer', 'X_Geneformer']
        }
    }

    data = {}

    for model_name, config in configs.items():
        file_path = None
        for fp in config['files']:
            if fp.exists():
                file_path = fp
                break

        if file_path is None:
            logger.warning(f"No file found for {model_name}")
            continue

        logger.info(f"Loading {model_name} from {file_path}")
        adata = sc.read_h5ad(file_path)

        # Filter to COVID cells with valid severity
        if model_name == 'scGPT':
            logger.info(f"  scGPT original shape: {adata.shape}")
            logger.info(f"  scGPT obs columns: {list(adata.obs.columns)}")

            # Try filtering by dataset column
            if 'dataset' in adata.obs.columns:
                logger.info(f"  dataset values: {adata.obs['dataset'].unique()}")
                covid_mask = adata.obs['dataset'] == 'covid'
                if covid_mask.sum() > 0:
                    adata = adata[covid_mask].copy()
                    logger.info(f"  After dataset filter: {adata.shape}")

            # Filter by valid severity labels
            if 'covid_severity' in adata.obs.columns:
                valid = ['healthy', 'mild', 'moderate', 'severe', 'fatal', 'recovered']
                # Convert to string and handle NaN
                severity_vals = adata.obs['covid_severity'].astype(str)
                logger.info(f"  severity values: {severity_vals.unique()[:10]}")
                mask = severity_vals.isin(valid)
                if mask.sum() > 0:
                    adata = adata[mask].copy()
                    logger.info(f"  After severity filter: {adata.shape}")
                else:
                    # Try lowercase
                    mask = severity_vals.str.lower().isin([v.lower() for v in valid])
                    if mask.sum() > 0:
                        adata = adata[mask].copy()
                        logger.info(f"  After lowercase severity filter: {adata.shape}")

        # Get embeddings
        obsm_key = None
        for key in config['obsm_keys']:
            if key in adata.obsm:
                obsm_key = key
                break

        if obsm_key is None:
            for k in adata.obsm.keys():
                if model_name.lower() in k.lower():
                    obsm_key = k
                    break

        if obsm_key is None:
            logger.warning(f"No embedding key found for {model_name}")
            continue

        embeddings = np.array(adata.obsm[obsm_key])

        # Get labels
        labels = None
        for col in ['covid_severity', 'severity', 'Category']:
            if col in adata.obs.columns:
                labels = np.array([str(x) if pd.notna(x) else 'unknown' for x in adata.obs[col]])
                break

        if labels is None:
            logger.warning(f"No labels found for {model_name}")
            continue

        # Filter to valid labels
        valid_mask = np.isin(labels, SEVERITY_ORDER)
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]

        data[model_name] = {
            'embeddings': embeddings,
            'labels': labels,
            'adata': adata[valid_mask].copy()
        }

        logger.info(f"  {model_name}: {embeddings.shape[0]} cells, {embeddings.shape[1]} dims")

    return data


def compute_dimensionality_reductions(embeddings, n_samples=10000, random_state=42):
    """Compute PCA, UMAP, t-SNE for embeddings"""
    np.random.seed(random_state)

    # Subsample if too large
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        X = embeddings[idx]
        sample_idx = idx
    else:
        X = embeddings
        sample_idx = np.arange(len(embeddings))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {'sample_idx': sample_idx}

    # PCA
    logger.info("  Computing PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    results['PCA'] = pca.fit_transform(X_scaled)
    results['pca_var'] = pca.explained_variance_ratio_

    # UMAP
    logger.info("  Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=30, min_dist=0.3)
    results['UMAP'] = reducer.fit_transform(X_scaled)

    # t-SNE
    logger.info("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    results['tSNE'] = tsne.fit_transform(X_scaled)

    return results


def get_binary_labels(labels):
    """Convert 6-class to binary labels"""
    binary = np.array(['non_severe'] * len(labels), dtype=object)
    binary[np.isin(labels, ['severe', 'fatal'])] = 'severe'
    return binary


def plot_single_embedding(coords, labels, title, ax, colors_dict, alpha=0.5, s=5, legend=True):
    """Plot a single embedding visualization"""
    unique_labels = [l for l in colors_dict.keys() if l in np.unique(labels)]

    for label in unique_labels:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors_dict[label], label=label, alpha=alpha, s=s, edgecolors='none')

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        ax.legend(loc='best', markerscale=3, fontsize=8)


def fig_umap_comparison_4models(data, output_dir, n_samples=10000):
    """
    Create UMAP comparison across all 4 models (2x2 grid)
    """
    logger.info("Creating UMAP comparison figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    model_order = ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']

    for idx, model_name in enumerate(model_order):
        if model_name not in data:
            axes[idx].text(0.5, 0.5, f'{model_name}\nNot Available',
                          ha='center', va='center', fontsize=14)
            axes[idx].set_axis_off()
            continue

        embeddings = data[model_name]['embeddings']
        labels = data[model_name]['labels']

        # Subsample
        np.random.seed(42)
        if len(embeddings) > n_samples:
            idx_sample = np.random.choice(len(embeddings), n_samples, replace=False)
            X = embeddings[idx_sample]
            y = labels[idx_sample]
        else:
            X = embeddings
            y = labels

        # Scale and compute UMAP
        X_scaled = StandardScaler().fit_transform(X)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(X_scaled)

        plot_single_embedding(coords, y, f'{model_name}\n({X.shape[1]} dims)',
                              axes[idx], SEVERITY_COLORS, alpha=0.6, s=8)

    plt.suptitle('UMAP Visualization: COVID-19 Severity by Foundation Model\n(6-class)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'umap_4models_6class.png')
    plt.savefig(output_dir / 'umap_4models_6class.pdf')
    plt.close()
    logger.info("Saved: umap_4models_6class.png/pdf")


def fig_umap_binary_4models(data, output_dir, n_samples=10000):
    """
    UMAP comparison with binary labels (severe vs non-severe)
    """
    logger.info("Creating UMAP binary comparison figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    model_order = ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']

    for idx, model_name in enumerate(model_order):
        if model_name not in data:
            axes[idx].text(0.5, 0.5, f'{model_name}\nNot Available',
                          ha='center', va='center', fontsize=14)
            axes[idx].set_axis_off()
            continue

        embeddings = data[model_name]['embeddings']
        labels = get_binary_labels(data[model_name]['labels'])

        # Subsample
        np.random.seed(42)
        if len(embeddings) > n_samples:
            idx_sample = np.random.choice(len(embeddings), n_samples, replace=False)
            X = embeddings[idx_sample]
            y = labels[idx_sample]
        else:
            X = embeddings
            y = labels

        # Scale and compute UMAP
        X_scaled = StandardScaler().fit_transform(X)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(X_scaled)

        plot_single_embedding(coords, y, f'{model_name}\n({X.shape[1]} dims)',
                              axes[idx], BINARY_COLORS, alpha=0.6, s=8)

    plt.suptitle('UMAP Visualization: COVID-19 Severity by Foundation Model\n(Binary: Severe vs Non-Severe)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'umap_4models_binary.png')
    plt.savefig(output_dir / 'umap_4models_binary.pdf')
    plt.close()
    logger.info("Saved: umap_4models_binary.png/pdf")


def fig_all_methods_single_model(data, model_name, output_dir, n_samples=8000):
    """
    PCA, UMAP, t-SNE for a single model side by side
    """
    if model_name not in data:
        logger.warning(f"{model_name} not in data")
        return

    logger.info(f"Creating all methods figure for {model_name}...")

    embeddings = data[model_name]['embeddings']
    labels = data[model_name]['labels']

    # Compute reductions
    results = compute_dimensionality_reductions(embeddings, n_samples=n_samples)
    y = labels[results['sample_idx']]

    # 6-class figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_single_embedding(results['PCA'], y,
                          f'PCA\n(Var: {results["pca_var"][0]:.1%}, {results["pca_var"][1]:.1%})',
                          axes[0], SEVERITY_COLORS, alpha=0.6, s=10)
    plot_single_embedding(results['UMAP'], y, 'UMAP',
                          axes[1], SEVERITY_COLORS, alpha=0.6, s=10)
    plot_single_embedding(results['tSNE'], y, 't-SNE',
                          axes[2], SEVERITY_COLORS, alpha=0.6, s=10)

    plt.suptitle(f'{model_name} Embeddings: Dimensionality Reduction Comparison\n(6-class COVID Severity)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f'{model_name.lower()}_pca_umap_tsne_6class.png')
    plt.savefig(output_dir / f'{model_name.lower()}_pca_umap_tsne_6class.pdf')
    plt.close()

    # Binary figure
    y_binary = get_binary_labels(y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_single_embedding(results['PCA'], y_binary,
                          f'PCA\n(Var: {results["pca_var"][0]:.1%}, {results["pca_var"][1]:.1%})',
                          axes[0], BINARY_COLORS, alpha=0.6, s=10)
    plot_single_embedding(results['UMAP'], y_binary, 'UMAP',
                          axes[1], BINARY_COLORS, alpha=0.6, s=10)
    plot_single_embedding(results['tSNE'], y_binary, 't-SNE',
                          axes[2], BINARY_COLORS, alpha=0.6, s=10)

    plt.suptitle(f'{model_name} Embeddings: Dimensionality Reduction Comparison\n(Binary: Severe vs Non-Severe)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_dir / f'{model_name.lower()}_pca_umap_tsne_binary.png')
    plt.savefig(output_dir / f'{model_name.lower()}_pca_umap_tsne_binary.pdf')
    plt.close()

    logger.info(f"Saved: {model_name.lower()}_pca_umap_tsne_*.png/pdf")


def fig_fancy_umap_with_density(data, model_name, output_dir, n_samples=10000):
    """
    Fancy UMAP with density contours and marginal distributions
    """
    if model_name not in data:
        return

    logger.info(f"Creating fancy UMAP for {model_name}...")

    embeddings = data[model_name]['embeddings']
    labels = data[model_name]['labels']

    # Subsample
    np.random.seed(42)
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels

    # Compute UMAP
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(X_scaled)

    # Create fancy figure with marginal distributions
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 4, figure=fig)

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])

    # Plot each severity level
    for severity in SEVERITY_ORDER:
        mask = y == severity
        if mask.sum() > 0:
            ax_main.scatter(coords[mask, 0], coords[mask, 1],
                           c=SEVERITY_COLORS[severity], label=severity,
                           alpha=0.6, s=15, edgecolors='none')

    ax_main.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax_main.legend(loc='lower right', markerscale=2)

    # Top marginal (UMAP 1 distribution)
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    for severity in SEVERITY_ORDER:
        mask = y == severity
        if mask.sum() > 0:
            ax_top.hist(coords[mask, 0], bins=50, alpha=0.5,
                       color=SEVERITY_COLORS[severity], density=True)
    ax_top.set_ylabel('Density')
    ax_top.tick_params(labelbottom=False)

    # Right marginal (UMAP 2 distribution)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    for severity in SEVERITY_ORDER:
        mask = y == severity
        if mask.sum() > 0:
            ax_right.hist(coords[mask, 1], bins=50, alpha=0.5,
                         color=SEVERITY_COLORS[severity], density=True, orientation='horizontal')
    ax_right.set_xlabel('Density')
    ax_right.tick_params(labelleft=False)

    plt.suptitle(f'{model_name}: UMAP with Marginal Distributions\n(COVID-19 Severity)',
                 fontsize=14, fontweight='bold')

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f'{model_name.lower()}_umap_fancy.png')
    plt.savefig(output_dir / f'{model_name.lower()}_umap_fancy.pdf')
    plt.close()
    logger.info(f"Saved: {model_name.lower()}_umap_fancy.png/pdf")


def fig_severity_progression(data, model_name, output_dir, n_samples=10000):
    """
    UMAP showing severity as a continuous progression (gradient coloring)
    """
    if model_name not in data:
        return

    logger.info(f"Creating severity progression figure for {model_name}...")

    embeddings = data[model_name]['embeddings']
    labels = data[model_name]['labels']

    # Create numeric severity score
    severity_score = {
        'healthy': 0, 'mild': 1, 'moderate': 2,
        'severe': 3, 'fatal': 4, 'recovered': 2.5  # recovered between moderate and severe
    }

    # Subsample
    np.random.seed(42)
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels

    scores = np.array([severity_score.get(l, 2) for l in y])

    # Compute UMAP
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(X_scaled)

    # Create custom colormap (green -> yellow -> red -> purple)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#8e44ad']
    cmap = LinearSegmentedColormap.from_list('severity', colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap=cmap,
                        alpha=0.7, s=15, edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Severity Score', fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Healthy', 'Mild', 'Moderate', 'Severe', 'Fatal'])

    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: COVID-19 Severity Progression\n(Continuous Gradient)',
                 fontsize=14, fontweight='bold')

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f'{model_name.lower()}_severity_gradient.png')
    plt.savefig(output_dir / f'{model_name.lower()}_severity_gradient.pdf')
    plt.close()
    logger.info(f"Saved: {model_name.lower()}_severity_gradient.png/pdf")


def fig_combined_mega_figure(data, output_dir, n_samples=8000):
    """
    Mega figure: 4 models x 3 methods (PCA, UMAP, t-SNE)
    """
    logger.info("Creating mega comparison figure...")

    model_order = ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']
    methods = ['PCA', 'UMAP', 'tSNE']

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    for row, model_name in enumerate(model_order):
        if model_name not in data:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, col].set_axis_off()
            continue

        embeddings = data[model_name]['embeddings']
        labels = data[model_name]['labels']

        # Compute reductions
        results = compute_dimensionality_reductions(embeddings, n_samples=n_samples)
        y = labels[results['sample_idx']]

        for col, method in enumerate(methods):
            coords = results[method]
            title = f'{model_name} - {method}'
            if method == 'PCA':
                title += f'\n(Var: {results["pca_var"][0]:.1%}+{results["pca_var"][1]:.1%})'

            plot_single_embedding(coords, y, title, axes[row, col],
                                  SEVERITY_COLORS, alpha=0.5, s=5, legend=(col==2))

    plt.suptitle('Foundation Model Embeddings: Comprehensive Visualization\n(COVID-19 6-class Severity)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'mega_comparison_4models_3methods.png')
    plt.savefig(output_dir / 'mega_comparison_4models_3methods.pdf')
    plt.close()
    logger.info("Saved: mega_comparison_4models_3methods.png/pdf")


def fig_silhouette_comparison(data, output_dir, n_samples=5000):
    """
    Compare clustering quality using silhouette scores
    """
    from sklearn.metrics import silhouette_score

    logger.info("Computing silhouette scores...")

    results = []

    for model_name in ['UCE', 'TranscriptFormer', 'Geneformer', 'scGPT']:
        if model_name not in data:
            continue

        embeddings = data[model_name]['embeddings']
        labels = data[model_name]['labels']

        # Subsample
        np.random.seed(42)
        if len(embeddings) > n_samples:
            idx = np.random.choice(len(embeddings), n_samples, replace=False)
            X = embeddings[idx]
            y = labels[idx]
        else:
            X = embeddings
            y = labels

        X_scaled = StandardScaler().fit_transform(X)

        # 6-class silhouette
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        sil_6class = silhouette_score(X_scaled, y_encoded)

        # Binary silhouette
        y_binary = get_binary_labels(y)
        y_binary_encoded = LabelEncoder().fit_transform(y_binary)
        sil_binary = silhouette_score(X_scaled, y_binary_encoded)

        results.append({
            'Model': model_name,
            '6-class': sil_6class,
            'Binary': sil_binary
        })

        logger.info(f"  {model_name}: 6-class={sil_6class:.3f}, binary={sil_binary:.3f}")

    df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['6-class'], width, label='6-class', color='#3498db')
    bars2 = ax.bar(x + width/2, df['Binary'], width, label='Binary', color='#e74c3c')

    ax.set_xlabel('Foundation Model', fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontweight='bold')
    ax.set_title('Embedding Quality: Silhouette Score by Model\n(Higher = Better Cluster Separation)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'])
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_dir = Path(output_dir)
    plt.savefig(output_dir / 'silhouette_comparison.png')
    plt.savefig(output_dir / 'silhouette_comparison.pdf')
    plt.close()
    logger.info("Saved: silhouette_comparison.png/pdf")

    # Save CSV
    df.to_csv(output_dir / 'silhouette_scores.csv', index=False)


def main():
    base_dir = Path("/bigdata/godziklab/shared/Xinru/302006")
    output_dir = base_dir / "05_RESULTS/covid_severity_benchmark/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings...")
    data = load_embeddings(base_dir)

    if not data:
        logger.error("No data loaded!")
        return 1

    logger.info(f"\nLoaded {len(data)} models: {list(data.keys())}")

    # Generate all figures
    logger.info("\n" + "="*60)
    logger.info("Generating visualizations...")
    logger.info("="*60)

    # 1. UMAP comparisons
    fig_umap_comparison_4models(data, output_dir)
    fig_umap_binary_4models(data, output_dir)

    # 2. All methods for each model (focus on UCE - best performer)
    fig_all_methods_single_model(data, 'UCE', output_dir)
    fig_all_methods_single_model(data, 'TranscriptFormer', output_dir)

    # 3. Fancy visualizations for UCE
    fig_fancy_umap_with_density(data, 'UCE', output_dir)
    fig_severity_progression(data, 'UCE', output_dir)

    # 4. Mega comparison figure
    fig_combined_mega_figure(data, output_dir)

    # 5. Silhouette score comparison
    fig_silhouette_comparison(data, output_dir)

    logger.info("\n" + "="*60)
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
