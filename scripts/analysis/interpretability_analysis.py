#!/usr/bin/env python3
"""
Interpretability Analysis

Extract and analyze LogReg coefficients to understand which embedding dimensions
drive severity predictions. Map back to biological meaning where possible.

Analysis:
1. Train LogReg on each model's embeddings
2. Extract top positive/negative coefficients
3. For raw expression baseline, identify top genes
4. Pathway enrichment analysis on top genes
5. Compare feature importance across models

Usage:
    python scripts/analysis/interpretability_analysis.py

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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class InterpretabilityAnalysis:
    """Analyze feature importance and interpretability of foundation model embeddings."""

    def __init__(self, base_dir="/bigdata/godziklab/shared/Xinru/302006"):
        self.base_dir = Path(base_dir)
        self.embedding_dir = self.base_dir / "02_EMBEDDINGS"
        self.output_dir = self.base_dir / "05_RESULTS/interpretability"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.model_configs = {
            'STATE': {
                'file': self.embedding_dir / "state/covid_state.h5ad",
                'obsm_key': ['X_state', 'X_STATE', 'X_emb'],
            },
            'UCE': {
                'file': self.embedding_dir / "uce/human_platelet_covid_severity_uce_adata.h5ad",
                'obsm_key': ['X_uce'],
            },
            'scGPT': {
                'file': self.embedding_dir / "scgpt/platelet_scgpt_combined.h5ad",
                'obsm_key': ['X_scGPT', 'X_scgpt'],
            },
            'Geneformer': {
                'file': self.embedding_dir / "geneformer/covid_geneformer.h5ad",
                'obsm_key': ['X_geneformer', 'X_Geneformer'],
            },
            'TranscriptFormer': {
                'file': self.embedding_dir / "transcriptformer/covid_transcriptformer.h5ad",
                'obsm_key': ['X_transcriptformer', 'X_TranscriptFormer'],
            },
        }

        # Severity mapping
        self.severity_to_binary = {
            'healthy': 'non_severe',
            'mild': 'non_severe',
            'moderate': 'non_severe',
            'recovered': 'non_severe',
            'severe': 'severe',
            'fatal': 'severe',
        }

    def load_embeddings(self, model_name):
        """Load embeddings for a specific model."""
        config = self.model_configs[model_name]

        if not config['file'].exists():
            logger.warning(f"File not found: {config['file']}")
            return None, None, None

        logger.info(f"Loading {model_name} from {config['file']}")
        adata = sc.read_h5ad(config['file'])

        # Find embedding key
        emb_key = None
        for key in config['obsm_key']:
            if key in adata.obsm:
                emb_key = key
                break

        if emb_key is None:
            logger.warning(f"No embedding key found for {model_name}")
            return None, None, None

        embeddings = np.array(adata.obsm[emb_key])
        logger.info(f"  Shape: {embeddings.shape}")

        return adata, embeddings, emb_key

    def load_raw_expression(self):
        """Load raw expression data for gene-level interpretability."""
        # Try to load from UCE file which has raw counts
        uce_file = self.embedding_dir / "uce/human_platelet_covid_severity_uce_adata.h5ad"

        if not uce_file.exists():
            logger.warning("Raw expression file not found")
            return None, None, None

        logger.info(f"Loading raw expression from {uce_file}")
        adata = sc.read_h5ad(uce_file)

        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)

        gene_names = list(adata.var_names)
        logger.info(f"  Expression shape: {X.shape}")
        logger.info(f"  Genes: {len(gene_names)}")

        return adata, X, gene_names

    def get_severity_labels(self, adata, task='binary'):
        """Extract severity labels."""
        severity_col = None
        for col in ['severity', 'disease_severity', 'covid_severity', 'condition']:
            if col in adata.obs.columns:
                severity_col = col
                break

        if severity_col is None:
            return None

        severity = adata.obs[severity_col].astype(str).str.lower()

        if task == 'binary':
            labels = severity.map(self.severity_to_binary)
        else:
            labels = severity

        return labels.values

    def train_logreg_and_get_coefficients(self, X, y, feature_names=None):
        """Train LogReg and extract coefficients."""
        # Filter valid samples
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Train LogReg with L2 regularization
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            C=1.0,  # Default regularization
            solver='lbfgs'
        )
        clf.fit(X_scaled, y_enc)

        # Get coefficients
        if len(le.classes_) == 2:
            # Binary: single coefficient vector
            coefficients = clf.coef_[0]
        else:
            # Multiclass: average absolute coefficients across classes
            coefficients = np.mean(np.abs(clf.coef_), axis=0)

        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_scaled, y_enc, cv=cv, scoring='roc_auc')

        # Create feature importance dataframe
        if feature_names is None:
            feature_names = [f"dim_{i}" for i in range(len(coefficients))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
        }).sort_values('abs_coefficient', ascending=False)

        return {
            'coefficients': coefficients,
            'importance_df': importance_df,
            'cv_auc': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'classes': list(le.classes_),
            'n_samples': len(y),
            'n_features': len(coefficients),
        }

    def analyze_model_interpretability(self, model_name, task='binary'):
        """Analyze interpretability for a specific model."""
        adata, embeddings, emb_key = self.load_embeddings(model_name)
        if embeddings is None:
            return None

        labels = self.get_severity_labels(adata, task)
        if labels is None:
            return None

        logger.info(f"  Training LogReg for {model_name}...")
        results = self.train_logreg_and_get_coefficients(embeddings, labels)
        results['model'] = model_name
        results['embedding_dim'] = embeddings.shape[1]

        logger.info(f"    CV AUC: {results['cv_auc']:.4f} ± {results['cv_std']:.4f}")
        logger.info(f"    Top 5 dimensions: {list(results['importance_df']['feature'].head())}")

        return results

    def analyze_raw_expression(self, task='binary', top_n=100):
        """Analyze raw expression for gene-level interpretability."""
        adata, X, gene_names = self.load_raw_expression()
        if X is None:
            return None

        labels = self.get_severity_labels(adata, task)
        if labels is None:
            return None

        # Use top variable genes for speed
        logger.info(f"  Selecting top {top_n} variable genes...")
        gene_var = np.var(X, axis=0)
        top_gene_idx = np.argsort(gene_var)[-top_n:]
        X_subset = X[:, top_gene_idx]
        gene_names_subset = [gene_names[i] for i in top_gene_idx]

        logger.info(f"  Training LogReg on raw expression ({top_n} genes)...")
        results = self.train_logreg_and_get_coefficients(
            X_subset, labels, feature_names=gene_names_subset
        )
        results['model'] = 'Raw_Expression'
        results['n_genes'] = top_n

        logger.info(f"    CV AUC: {results['cv_auc']:.4f} ± {results['cv_std']:.4f}")

        return results

    def run_full_analysis(self, models=None, task='binary'):
        """Run interpretability analysis for all models."""
        if models is None:
            models = list(self.model_configs.keys())

        all_results = {}

        # Analyze each foundation model
        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'='*60}")

            results = self.analyze_model_interpretability(model_name, task)
            if results:
                all_results[model_name] = results

        # Analyze raw expression
        logger.info(f"\n{'='*60}")
        logger.info(f"Raw Expression Analysis")
        logger.info(f"{'='*60}")

        raw_results = self.analyze_raw_expression(task, top_n=500)
        if raw_results:
            all_results['Raw_Expression'] = raw_results

        return all_results

    def create_visualizations(self, results, output_prefix):
        """Create interpretability visualizations."""

        # 1. Coefficient distribution comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        model_colors = {
            'STATE': '#e74c3c',
            'UCE': '#3498db',
            'scGPT': '#2ecc71',
            'Geneformer': '#9b59b6',
            'TranscriptFormer': '#f39c12',
            'Raw_Expression': '#34495e',
        }

        for idx, (model_name, model_results) in enumerate(results.items()):
            if idx >= 6:
                break
            ax = axes[idx]
            coeffs = model_results['coefficients']
            color = model_colors.get(model_name, '#95a5a6')

            ax.hist(coeffs, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f"{model_name}\n(dim={model_results.get('embedding_dim', model_results.get('n_genes', 'N/A'))})",
                        fontweight='bold')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Frequency')

            # Add stats
            stats_text = f"Mean: {np.mean(coeffs):.3f}\nStd: {np.std(coeffs):.3f}\nAUC: {model_results['cv_auc']:.3f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

        plt.suptitle('LogReg Coefficient Distributions by Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_coefficient_distributions.png", bbox_inches='tight', facecolor='white')
        plt.savefig(f"{output_prefix}_coefficient_distributions.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_prefix}_coefficient_distributions.png")

        # 2. Top features comparison
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (model_name, model_results) in enumerate(results.items()):
            if idx >= 6:
                break
            ax = axes[idx]
            top_df = model_results['importance_df'].head(15)
            color = model_colors.get(model_name, '#95a5a6')

            # Color bars by sign of coefficient
            colors = [color if c > 0 else '#7f8c8d' for c in top_df['coefficient']]

            bars = ax.barh(range(len(top_df)), top_df['coefficient'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(top_df)))
            ax.set_yticklabels(top_df['feature'], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title(f"{model_name} - Top 15 Features", fontweight='bold')
            ax.set_xlabel('Coefficient')

        plt.suptitle('Top Predictive Features by Model\n(Positive = predicts severe, Negative = predicts non-severe)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_top_features.png", bbox_inches='tight', facecolor='white')
        plt.savefig(f"{output_prefix}_top_features.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_prefix}_top_features.png")

        # 3. Model comparison summary
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # AUC comparison
        ax1 = axes[0]
        models_list = list(results.keys())
        aucs = [results[m]['cv_auc'] for m in models_list]
        stds = [results[m]['cv_std'] for m in models_list]
        colors = [model_colors.get(m, '#95a5a6') for m in models_list]

        bars = ax1.bar(models_list, aucs, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax1.set_ylabel('CV AUC')
        ax1.set_title('Model Performance (5-Fold CV)', fontweight='bold')
        ax1.set_xticklabels(models_list, rotation=45, ha='right')
        ax1.set_ylim(0.5, 1.0)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Add value labels
        for bar, auc in zip(bars, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=9)

        # Coefficient sparsity (% of coefficients > threshold)
        ax2 = axes[1]
        threshold = 0.1
        sparsity = []
        for m in models_list:
            coeffs = results[m]['coefficients']
            pct_important = np.mean(np.abs(coeffs) > threshold) * 100
            sparsity.append(pct_important)

        bars = ax2.bar(models_list, sparsity, color=colors, alpha=0.8)
        ax2.set_ylabel(f'% Features with |coef| > {threshold}')
        ax2.set_title('Feature Importance Concentration', fontweight='bold')
        ax2.set_xticklabels(models_list, rotation=45, ha='right')

        for bar, s in zip(bars, sparsity):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{s:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Interpretability Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_summary.png", bbox_inches='tight', facecolor='white')
        plt.savefig(f"{output_prefix}_summary.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_prefix}_summary.png")

    def save_results(self, results, task='binary'):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save top genes from raw expression
        if 'Raw_Expression' in results:
            raw_df = results['Raw_Expression']['importance_df']
            raw_file = self.output_dir / f"top_genes_raw_expression_{task}_{timestamp}.csv"
            raw_df.to_csv(raw_file, index=False)
            logger.info(f"Saved: {raw_file}")

        # Save top features for each model
        for model_name, model_results in results.items():
            df = model_results['importance_df']
            csv_file = self.output_dir / f"top_features_{model_name}_{task}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

        # Save summary
        summary_data = []
        for model_name, model_results in results.items():
            summary_data.append({
                'model': model_name,
                'cv_auc': model_results['cv_auc'],
                'cv_std': model_results['cv_std'],
                'n_features': model_results['n_features'],
                'n_samples': model_results['n_samples'],
                'mean_coef': np.mean(model_results['coefficients']),
                'std_coef': np.std(model_results['coefficients']),
                'max_coef': np.max(model_results['coefficients']),
                'min_coef': np.min(model_results['coefficients']),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"interpretability_summary_{task}_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved: {summary_file}")

        # Save full results as JSON (excluding large arrays)
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'cv_auc': model_results['cv_auc'],
                'cv_std': model_results['cv_std'],
                'n_features': model_results['n_features'],
                'n_samples': model_results['n_samples'],
                'top_10_features': model_results['importance_df'].head(10).to_dict('records'),
            }

        json_file = self.output_dir / f"interpretability_results_{task}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved: {json_file}")

        return summary_file

    def extract_top_genes_analysis(self, results, top_n=50):
        """Extract and analyze top genes from raw expression."""
        if 'Raw_Expression' not in results:
            logger.warning("No raw expression results available")
            return None

        raw_df = results['Raw_Expression']['importance_df']

        # Get top genes predicting severe
        top_severe = raw_df[raw_df['coefficient'] > 0].head(top_n)

        # Get top genes predicting non-severe (protective)
        top_protective = raw_df[raw_df['coefficient'] < 0].head(top_n)

        logger.info(f"\nTop {top_n} genes predicting SEVERE:")
        logger.info("-" * 50)
        for i, row in top_severe.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

        logger.info(f"\nTop {top_n} genes predicting NON-SEVERE (protective):")
        logger.info("-" * 50)
        for i, row in top_protective.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

        return {
            'severe_genes': list(top_severe['feature']),
            'protective_genes': list(top_protective['feature']),
        }


def main():
    logger.info("=" * 80)
    logger.info("INTERPRETABILITY ANALYSIS")
    logger.info("=" * 80)

    analyzer = InterpretabilityAnalysis()

    # Run analysis for binary task
    logger.info("\n" + "=" * 80)
    logger.info("BINARY CLASSIFICATION (Severe vs Non-Severe)")
    logger.info("=" * 80)

    results_binary = analyzer.run_full_analysis(task='binary')

    # Create visualizations
    output_prefix = str(analyzer.output_dir / "interpretability_binary")
    analyzer.create_visualizations(results_binary, output_prefix)

    # Save results
    analyzer.save_results(results_binary, task='binary')

    # Extract top genes
    gene_analysis = analyzer.extract_top_genes_analysis(results_binary)

    # Run analysis for 3-class task
    logger.info("\n" + "=" * 80)
    logger.info("3-CLASS CLASSIFICATION (Control vs Mild vs Severe)")
    logger.info("=" * 80)

    results_3class = analyzer.run_full_analysis(task='3-class')

    output_prefix_3class = str(analyzer.output_dir / "interpretability_3class")
    analyzer.create_visualizations(results_3class, output_prefix_3class)
    analyzer.save_results(results_3class, task='3-class')

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    print("\nBinary Classification - Model Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'CV AUC':<12} {'Dimensions':<12} {'Sparsity':<12}")
    print("-" * 60)

    for model_name, model_results in results_binary.items():
        n_dim = model_results.get('embedding_dim', model_results.get('n_genes', 'N/A'))
        coeffs = model_results['coefficients']
        sparsity = np.mean(np.abs(coeffs) > 0.1) * 100
        print(f"{model_name:<20} {model_results['cv_auc']:.4f} ± {model_results['cv_std']:.4f}  {str(n_dim):<12} {sparsity:.1f}%")

    if gene_analysis:
        print("\n" + "=" * 60)
        print("TOP GENES FROM RAW EXPRESSION")
        print("=" * 60)
        print("\nTop 10 genes predicting SEVERE:")
        for gene in gene_analysis['severe_genes'][:10]:
            print(f"  - {gene}")
        print("\nTop 10 genes predicting NON-SEVERE:")
        for gene in gene_analysis['protective_genes'][:10]:
            print(f"  - {gene}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
