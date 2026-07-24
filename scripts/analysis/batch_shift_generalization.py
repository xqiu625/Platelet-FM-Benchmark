#!/usr/bin/env python3
"""
Batch-Shift Generalization Analysis

Test if foundation model embeddings generalize across different diseases/batches.
- Train on COVID-19 → Test on Sepsis
- Train on Sepsis → Test on COVID-19

This tests the key promise of foundation models: generalizable representations.

Usage:
    python scripts/analysis/batch_shift_generalization.py

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchShiftAnalysis:
    """Analyze cross-disease generalization of foundation model embeddings."""

    def __init__(self, base_dir="/bigdata/godziklab/shared/Xinru/302006"):
        self.base_dir = Path(base_dir)
        self.embedding_dir = self.base_dir / "02_EMBEDDINGS"
        self.output_dir = self.base_dir / "05_RESULTS/covid_severity_benchmark"
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

        # Severity mapping to binary
        self.severity_to_binary = {
            'healthy': 'non_severe',
            'mild': 'non_severe',
            'moderate': 'non_severe',
            'recovered': 'non_severe',
            'severe': 'severe',
            'fatal': 'severe',
            # Sepsis categories
            'sepsis_mild': 'non_severe',
            'sepsis_severe': 'severe',
            'control': 'non_severe',
        }

    def load_embeddings(self, model_name):
        """Load embeddings for a specific model."""
        config = self.model_configs[model_name]

        if not config['file'].exists():
            logger.warning(f"File not found: {config['file']}")
            return None, None

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
            return None, None

        embeddings = np.array(adata.obsm[emb_key])
        logger.info(f"  Shape: {embeddings.shape}")

        return adata, embeddings

    def get_disease_labels(self, adata):
        """Extract disease type (COVID vs Sepsis) and severity labels."""
        # Try different column names
        severity_col = None
        for col in ['severity', 'disease_severity', 'covid_severity', 'condition']:
            if col in adata.obs.columns:
                severity_col = col
                break

        if severity_col is None:
            logger.warning("No severity column found")
            return None, None, None

        severity = adata.obs[severity_col].astype(str).str.lower()

        # Determine disease type
        disease_type = []
        for s in severity:
            if 'sepsis' in s:
                disease_type.append('sepsis')
            elif s in ['healthy', 'mild', 'moderate', 'severe', 'fatal', 'recovered']:
                disease_type.append('covid')
            else:
                disease_type.append('unknown')

        disease_type = np.array(disease_type)

        # Map to binary severity
        binary_severity = severity.map(self.severity_to_binary)

        return severity.values, binary_severity.values, disease_type

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, classifier='logreg'):
        """Train classifier and evaluate on test set."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # Train classifier
        if classifier == 'logreg':
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        clf.fit(X_train_scaled, y_train_enc)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test_enc, y_pred)
        balanced_acc = balanced_accuracy_score(y_test_enc, y_pred)

        # AUC (handle binary vs multiclass)
        if len(le.classes_) == 2:
            auc = roc_auc_score(y_test_enc, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test_enc, y_prob, multi_class='ovr', average='macro')

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'auc': auc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'classes': list(le.classes_),
            'confusion_matrix': confusion_matrix(y_test_enc, y_pred).tolist(),
        }

    def run_cross_disease_analysis(self, models=None):
        """Run cross-disease generalization analysis."""
        if models is None:
            models = list(self.model_configs.keys())

        results = {
            'covid_to_sepsis': {},
            'sepsis_to_covid': {},
            'within_covid': {},
            'within_sepsis': {},
        }

        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'='*60}")

            adata, embeddings = self.load_embeddings(model_name)
            if embeddings is None:
                continue

            severity, binary_severity, disease_type = self.get_disease_labels(adata)
            if severity is None:
                continue

            # Filter out unknown disease types and NaN severities
            valid_mask = (disease_type != 'unknown') & (~pd.isna(binary_severity))
            embeddings = embeddings[valid_mask]
            severity = severity[valid_mask]
            binary_severity = binary_severity[valid_mask]
            disease_type = disease_type[valid_mask]

            # Check if we have both COVID and Sepsis
            covid_mask = disease_type == 'covid'
            sepsis_mask = disease_type == 'sepsis'

            n_covid = covid_mask.sum()
            n_sepsis = sepsis_mask.sum()

            logger.info(f"  COVID samples: {n_covid}")
            logger.info(f"  Sepsis samples: {n_sepsis}")

            if n_covid < 100:
                logger.warning(f"  Not enough COVID samples, skipping")
                continue

            # 1. Within-COVID (baseline)
            logger.info("\n  [Within-COVID Training/Testing]")
            try:
                # Use 80/20 split within COVID
                np.random.seed(42)
                covid_indices = np.where(covid_mask)[0]
                np.random.shuffle(covid_indices)
                split = int(0.8 * len(covid_indices))
                train_idx = covid_indices[:split]
                test_idx = covid_indices[split:]

                result = self.train_and_evaluate(
                    embeddings[train_idx], binary_severity[train_idx],
                    embeddings[test_idx], binary_severity[test_idx]
                )
                results['within_covid'][model_name] = result
                logger.info(f"    AUC: {result['auc']:.4f} (train={result['n_train']}, test={result['n_test']})")
            except Exception as e:
                logger.error(f"    Error: {e}")

            # Only do cross-disease if we have sepsis samples
            if n_sepsis >= 50:
                # 2. COVID → Sepsis
                logger.info("\n  [COVID → Sepsis Transfer]")
                try:
                    result = self.train_and_evaluate(
                        embeddings[covid_mask], binary_severity[covid_mask],
                        embeddings[sepsis_mask], binary_severity[sepsis_mask]
                    )
                    results['covid_to_sepsis'][model_name] = result
                    logger.info(f"    AUC: {result['auc']:.4f} (train={result['n_train']}, test={result['n_test']})")
                except Exception as e:
                    logger.error(f"    Error: {e}")

                # 3. Sepsis → COVID
                logger.info("\n  [Sepsis → COVID Transfer]")
                try:
                    result = self.train_and_evaluate(
                        embeddings[sepsis_mask], binary_severity[sepsis_mask],
                        embeddings[covid_mask], binary_severity[covid_mask]
                    )
                    results['sepsis_to_covid'][model_name] = result
                    logger.info(f"    AUC: {result['auc']:.4f} (train={result['n_train']}, test={result['n_test']})")
                except Exception as e:
                    logger.error(f"    Error: {e}")

                # 4. Within-Sepsis (if enough samples)
                if n_sepsis >= 100:
                    logger.info("\n  [Within-Sepsis Training/Testing]")
                    try:
                        sepsis_indices = np.where(sepsis_mask)[0]
                        np.random.shuffle(sepsis_indices)
                        split = int(0.8 * len(sepsis_indices))
                        train_idx = sepsis_indices[:split]
                        test_idx = sepsis_indices[split:]

                        result = self.train_and_evaluate(
                            embeddings[train_idx], binary_severity[train_idx],
                            embeddings[test_idx], binary_severity[test_idx]
                        )
                        results['within_sepsis'][model_name] = result
                        logger.info(f"    AUC: {result['auc']:.4f} (train={result['n_train']}, test={result['n_test']})")
                    except Exception as e:
                        logger.error(f"    Error: {e}")
            else:
                logger.info(f"  Skipping cross-disease (only {n_sepsis} sepsis samples)")

        return results

    def run_batch_effect_analysis(self, models=None):
        """
        Alternative: Test generalization across different batches/datasets within COVID.
        Split by sample/patient to test batch generalization.
        """
        if models is None:
            models = list(self.model_configs.keys())

        results = {}

        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name} - Batch Effect Analysis")
            logger.info(f"{'='*60}")

            adata, embeddings = self.load_embeddings(model_name)
            if embeddings is None:
                continue

            severity, binary_severity, disease_type = self.get_disease_labels(adata)
            if severity is None:
                continue

            # Check for sample/batch column
            # Data_NO contains the 12 data sources (e.g., 302004data01, 302005data02, etc.)
            batch_col = None
            for col in ['Data_NO', 'data_no', 'data_source', 'sample', 'batch', 'patient', 'donor', 'dataset']:
                if col in adata.obs.columns:
                    batch_col = col
                    break

            if batch_col is None:
                logger.warning(f"  No batch column found, skipping")
                continue

            # Convert to string to handle mixed types (float/string)
            batches = adata.obs[batch_col].astype(str).values
            unique_batches = np.unique(batches)
            # Filter out 'nan' string values
            unique_batches = [b for b in unique_batches if b.lower() != 'nan']
            logger.info(f"  Found {len(unique_batches)} unique batches in '{batch_col}'")

            # Filter valid samples
            valid_mask = ~pd.isna(binary_severity)
            embeddings = embeddings[valid_mask]
            binary_severity = binary_severity[valid_mask]
            batches = batches[valid_mask]

            # Leave-one-batch-out cross-validation
            batch_results = []

            for test_batch in unique_batches:  # Test all 12 data sources
                train_mask = batches != test_batch
                test_mask = batches == test_batch

                if train_mask.sum() < 100 or test_mask.sum() < 20:
                    continue

                try:
                    result = self.train_and_evaluate(
                        embeddings[train_mask], binary_severity[train_mask],
                        embeddings[test_mask], binary_severity[test_mask]
                    )
                    batch_results.append({
                        'test_batch': str(test_batch),
                        **result
                    })
                    logger.info(f"    Batch '{test_batch}': AUC={result['auc']:.4f} (n_test={result['n_test']})")
                except Exception as e:
                    logger.warning(f"    Batch '{test_batch}': Error - {e}")

            if batch_results:
                # Filter out NaN AUC values
                aucs = [r['auc'] for r in batch_results if not np.isnan(r['auc'])]
                if aucs:
                    results[model_name] = {
                        'mean_auc': np.mean(aucs),
                        'std_auc': np.std(aucs),
                        'min_auc': np.min(aucs),
                        'max_auc': np.max(aucs),
                        'n_batches': len(aucs),
                        'n_batches_total': len(batch_results),
                        'batch_results': batch_results,
                    }
                    logger.info(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} ({len(aucs)} valid batches)")

        return results

    def create_visualization(self, results, output_path):
        """Create visualization of generalization results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Prepare data for plotting
        models = []
        within_covid = []
        covid_to_sepsis = []
        sepsis_to_covid = []

        for model in self.model_configs.keys():
            if model in results['within_covid']:
                models.append(model)
                within_covid.append(results['within_covid'][model]['auc'])
                covid_to_sepsis.append(results['covid_to_sepsis'].get(model, {}).get('auc', 0))
                sepsis_to_covid.append(results['sepsis_to_covid'].get(model, {}).get('auc', 0))

        if not models:
            logger.warning("No results to visualize")
            return

        x = np.arange(len(models))
        width = 0.25

        # Plot 1: Cross-disease comparison
        ax1 = axes[0]
        bars1 = ax1.bar(x - width, within_covid, width, label='Within-COVID', color='#3498db')
        bars2 = ax1.bar(x, covid_to_sepsis, width, label='COVID→Sepsis', color='#e74c3c')
        bars3 = ax1.bar(x + width, sepsis_to_covid, width, label='Sepsis→COVID', color='#2ecc71')

        ax1.set_ylabel('AUC')
        ax1.set_title('Cross-Disease Generalization', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0.5, 1.0)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                if bar.get_height() > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Transfer gap
        ax2 = axes[1]
        transfer_gap = [w - c for w, c in zip(within_covid, covid_to_sepsis)]
        colors = ['#e74c3c' if g > 0.1 else '#f39c12' if g > 0.05 else '#2ecc71' for g in transfer_gap]

        bars = ax2.bar(models, transfer_gap, color=colors)
        ax2.set_ylabel('AUC Drop (Within - Transfer)')
        ax2.set_title('Transfer Gap (smaller = better generalization)', fontweight='bold')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax2.legend()

        # Add value labels
        for bar, val in zip(bars, transfer_gap):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Foundation Model Generalization: COVID ↔ Sepsis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def save_results(self, results, batch_results=None):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save cross-disease results
        output_file = self.output_dir / f"batch_shift_generalization_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved: {output_file}")

        # Create summary CSV
        summary_data = []
        for model in self.model_configs.keys():
            row = {'model': model}
            for scenario in ['within_covid', 'covid_to_sepsis', 'sepsis_to_covid', 'within_sepsis']:
                if model in results.get(scenario, {}):
                    row[f'{scenario}_auc'] = results[scenario][model]['auc']
                    row[f'{scenario}_n_train'] = results[scenario][model]['n_train']
                    row[f'{scenario}_n_test'] = results[scenario][model]['n_test']
            if row:
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_dir / f"batch_shift_generalization_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved: {csv_file}")

        # Save batch results if available
        if batch_results:
            batch_file = self.output_dir / f"batch_effect_analysis_{timestamp}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            logger.info(f"Saved: {batch_file}")

        return output_file


def main():
    logger.info("=" * 80)
    logger.info("BATCH-SHIFT GENERALIZATION ANALYSIS")
    logger.info("=" * 80)

    analyzer = BatchShiftAnalysis()

    # Run cross-disease analysis
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: Cross-Disease Generalization (COVID ↔ Sepsis)")
    logger.info("=" * 80)
    results = analyzer.run_cross_disease_analysis()

    # Run batch effect analysis
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: Batch Effect Analysis (Leave-One-Batch-Out)")
    logger.info("=" * 80)
    batch_results = analyzer.run_batch_effect_analysis()

    # Save results
    output_file = analyzer.save_results(results, batch_results)

    # Create visualization
    viz_path = str(analyzer.output_dir / "batch_shift_generalization.png")
    analyzer.create_visualization(results, viz_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    print("\nCross-Disease Generalization (AUC):")
    print("-" * 70)
    print(f"{'Model':<20} {'Within-COVID':<15} {'COVID→Sepsis':<15} {'Gap':<10}")
    print("-" * 70)

    for model in analyzer.model_configs.keys():
        within = results['within_covid'].get(model, {}).get('auc', 0)
        transfer = results['covid_to_sepsis'].get(model, {}).get('auc', 0)
        gap = within - transfer if within and transfer else 0
        print(f"{model:<20} {within:<15.4f} {transfer:<15.4f} {gap:<10.4f}")

    if batch_results:
        print("\nBatch Effect Analysis (Leave-One-Batch-Out):")
        print("-" * 50)
        print(f"{'Model':<20} {'Mean AUC':<15} {'Std':<10}")
        print("-" * 50)
        for model, res in batch_results.items():
            print(f"{model:<20} {res['mean_auc']:<15.4f} {res['std_auc']:<10.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
