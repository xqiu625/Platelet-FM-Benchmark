#!/usr/bin/env python3
"""
Gene Dropout Robustness Analysis for Foundation Model Benchmark

Tests how model performance degrades when genes are randomly dropped:
- 0%, 10%, 30%, 50%, 70% gene dropout
- Simulates scRNA-seq dropout and missing data scenarios
- Measures AUC at each dropout level

Usage:
    python gene_dropout_robustness.py --tasks binary 3-class
    python gene_dropout_robustness.py --tasks binary --n-repeats 5

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

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from datetime import datetime
import json
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_embedding_config(model_name, base_dir):
    """Get configuration for a specific model"""
    embedding_dir = Path(base_dir) / "02_EMBEDDINGS"

    configs = {
        'UCE': {
            'files': [
                embedding_dir / "uce/human_platelet_covid_severity_uce_adata.h5ad",
            ],
            'obsm_keys': ['X_uce'],
        },
        'scGPT': {
            'files': [
                embedding_dir / "scgpt/covid_scgpt.h5ad",
                embedding_dir / "scgpt/platelet_scgpt_combined.h5ad",
            ],
            'obsm_keys': ['X_scGPT', 'X_scgpt'],
        },
        'TranscriptFormer': {
            'files': [
                embedding_dir / "transcriptformer/covid_transcriptformer.h5ad",
            ],
            'obsm_keys': ['X_transcriptformer', 'X_TranscriptFormer'],
        },
        'Geneformer': {
            'files': [
                embedding_dir / "geneformer/covid_geneformer.h5ad",
            ],
            'obsm_keys': ['X_geneformer', 'X_Geneformer'],
        },
        'STATE': {
            'files': [
                embedding_dir / "state/covid_state.h5ad",
            ],
            'obsm_keys': ['X_emb', 'X_state'],
        }
    }

    return configs.get(model_name)


def load_model_data(model_name, base_dir):
    """Load embeddings and labels for a model"""
    config = get_embedding_config(model_name, base_dir)

    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    file_path = None
    for fp in config['files']:
        if fp.exists():
            file_path = fp
            break

    if file_path is None:
        logger.warning(f"No file found for {model_name}")
        return None, None

    logger.info(f"Loading {model_name} from {file_path}")
    adata = sc.read_h5ad(file_path)

    # Filter to COVID only for scGPT combined file
    if model_name == 'scGPT' and 'platelet_scgpt_combined' in str(file_path):
        if 'dataset' in adata.obs.columns:
            covid_mask = adata.obs['dataset'] == 'covid'
            if covid_mask.sum() > 0:
                adata = adata[covid_mask].copy()

    # Get embeddings
    obsm_key = None
    for key in config['obsm_keys']:
        if key in adata.obsm:
            obsm_key = key
            break

    if obsm_key is None:
        logger.warning(f"No embedding key found for {model_name}")
        return None, None

    embeddings = adata.obsm[obsm_key]
    if hasattr(embeddings, 'toarray'):
        embeddings = embeddings.toarray()
    embeddings = np.array(embeddings)

    # Get labels
    labels = None
    for col in ['covid_severity', 'severity', 'Category']:
        if col in adata.obs.columns:
            raw_labels = adata.obs[col].values
            labels = np.array([str(x) if pd.notna(x) else 'unknown' for x in raw_labels])
            break

    return embeddings, labels


def prepare_labels(task_name, labels):
    """Prepare labels for classification task"""
    valid_severities = ['healthy', 'mild', 'moderate', 'severe', 'fatal', 'recovered']

    tasks = {
        '6-class': None,
        '3-class': {
            'control': ['healthy'],
            'mild': ['mild', 'moderate', 'recovered'],
            'severe': ['severe', 'fatal']
        },
        'binary': {
            'non_severe': ['healthy', 'mild', 'moderate', 'recovered'],
            'severe': ['severe', 'fatal']
        }
    }

    if task_name == '6-class':
        valid_mask = np.isin(labels, valid_severities)
        return labels, valid_mask

    task_mapping = tasks[task_name]
    new_labels = np.array(['unknown'] * len(labels), dtype=object)

    for new_label, old_labels in task_mapping.items():
        for old_label in old_labels:
            mask = labels == old_label
            new_labels[mask] = new_label

    valid_mask = new_labels != 'unknown'
    return new_labels, valid_mask


def apply_gene_dropout(X, dropout_frac, random_state=42):
    """
    Apply gene (dimension) dropout to embeddings.
    Sets a fraction of dimensions to zero.
    """
    if dropout_frac == 0:
        return X.copy()

    np.random.seed(random_state)
    X_dropped = X.copy()
    n_dims = X.shape[1]
    n_drop = int(n_dims * dropout_frac)

    # Randomly select dimensions to drop
    drop_indices = np.random.choice(n_dims, n_drop, replace=False)
    X_dropped[:, drop_indices] = 0

    return X_dropped


def compute_auc(y_true, y_prob, n_classes):
    """Compute AUC for binary or multi-class"""
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')


def evaluate_with_dropout(X, y, dropout_frac, n_repeats=5, n_splits=5):
    """Evaluate model with gene dropout"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    aucs = []
    balanced_accs = []

    for repeat in range(n_repeats):
        seed = 42 + repeat

        # Apply dropout
        X_dropped = apply_gene_dropout(X, dropout_frac, random_state=seed)

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=2000, random_state=seed, n_jobs=-1,
                class_weight='balanced', solver='saga'
            ))
        ])

        try:
            y_pred = cross_val_predict(pipeline, X_dropped, y_encoded, cv=cv)
            y_prob = cross_val_predict(pipeline, X_dropped, y_encoded, cv=cv, method='predict_proba')

            auc = compute_auc(y_encoded, y_prob, n_classes)
            bal_acc = balanced_accuracy_score(y_encoded, y_pred)

            aucs.append(auc)
            balanced_accs.append(bal_acc)
        except Exception as e:
            logger.warning(f"  Repeat {repeat+1} failed: {e}")

    if not aucs:
        return None

    return {
        'dropout_frac': dropout_frac,
        'dropout_pct': int(dropout_frac * 100),
        'n_dims_dropped': int(X.shape[1] * dropout_frac),
        'n_dims_remaining': int(X.shape[1] * (1 - dropout_frac)),
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'auc_min': np.min(aucs),
        'auc_max': np.max(aucs),
        'balanced_acc_mean': np.mean(balanced_accs),
        'balanced_acc_std': np.std(balanced_accs),
        'n_repeats': len(aucs)
    }


def run_gene_dropout_analysis(base_dir, tasks=None, n_repeats=5, dropout_fracs=None):
    """Run gene dropout robustness analysis"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['binary', '3-class']

    if dropout_fracs is None:
        dropout_fracs = [0.0, 0.1, 0.3, 0.5, 0.7]

    models = ['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer']

    logger.info("="*80)
    logger.info("GENE DROPOUT ROBUSTNESS ANALYSIS")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Dropout fractions: {dropout_fracs}")
    logger.info(f"Repeats per fraction: {n_repeats}")
    logger.info("="*80)

    all_results = {}

    for task_name in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_name}")
        logger.info(f"{'='*60}")

        task_results = {}

        for model_name in models:
            logger.info(f"\n--- {model_name} ---")

            try:
                X, labels = load_model_data(model_name, base_dir)
                if X is None:
                    logger.warning(f"  Skipping {model_name} - no data")
                    continue

                y, valid_mask = prepare_labels(task_name, labels)
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]

                logger.info(f"  Total samples: {len(y_valid)}, Embedding dims: {X_valid.shape[1]}")

                model_results = []

                for dropout_frac in dropout_fracs:
                    logger.info(f"  Dropout {dropout_frac*100:.0f}%...")

                    result = evaluate_with_dropout(
                        X_valid, y_valid,
                        dropout_frac=dropout_frac,
                        n_repeats=n_repeats
                    )

                    if result:
                        result['model'] = model_name
                        result['embedding_dims'] = X_valid.shape[1]
                        model_results.append(result)
                        logger.info(f"    AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f} "
                                   f"(dims remaining: {result['n_dims_remaining']})")

                task_results[model_name] = model_results

            except Exception as e:
                logger.error(f"  Error: {str(e)}")

        all_results[task_name] = task_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_file = output_dir / f'gene_dropout_robustness_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_file}")

    # CSV
    rows = []
    for task_name, task_results in all_results.items():
        for model_name, model_results in task_results.items():
            for result in model_results:
                rows.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Dropout_Frac': result['dropout_frac'],
                    'Dropout_Pct': result['dropout_pct'],
                    'Embedding_Dims': result['embedding_dims'],
                    'Dims_Dropped': result['n_dims_dropped'],
                    'Dims_Remaining': result['n_dims_remaining'],
                    'AUC_Mean': result['auc_mean'],
                    'AUC_Std': result['auc_std'],
                    'AUC_Min': result['auc_min'],
                    'AUC_Max': result['auc_max'],
                    'BalancedAcc_Mean': result['balanced_acc_mean'],
                    'BalancedAcc_Std': result['balanced_acc_std'],
                    'N_Repeats': result['n_repeats']
                })

    df = pd.DataFrame(rows)
    csv_file = output_dir / f'gene_dropout_robustness_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("GENE DROPOUT ROBUSTNESS SUMMARY")
    print(f"{'='*80}")

    for task_name in all_results.keys():
        print(f"\n{task_name}:")
        print("-" * 80)
        print(f"{'Model':<18} {'0%':>10} {'10%':>10} {'30%':>10} {'50%':>10} {'70%':>10}")
        print("-" * 80)

        for model_name in models:
            if model_name in all_results[task_name]:
                results = all_results[task_name][model_name]
                values = {r['dropout_frac']: r['auc_mean'] for r in results}

                row = f"{model_name:<18}"
                for frac in [0.0, 0.1, 0.3, 0.5, 0.7]:
                    if frac in values:
                        row += f" {values[frac]:>9.4f}"
                    else:
                        row += f" {'N/A':>9}"
                print(row)

    print(f"\n{'='*80}")
    print("Interpretation:")
    print("  - Flat curve = robust to dimension dropout")
    print("  - Steep drop = sensitive to missing dimensions")
    print("  - Models with redundant representations are more robust")
    print(f"{'='*80}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Gene Dropout Robustness Analysis')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory')
    parser.add_argument('--tasks', nargs='+', default=['binary', '3-class'],
                        choices=['binary', '3-class', '6-class'],
                        help='Tasks to analyze')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Number of repeats per dropout level')
    parser.add_argument('--dropout-fracs', nargs='+', type=float,
                        default=[0.0, 0.1, 0.3, 0.5, 0.7],
                        help='Dropout fractions to test')
    args = parser.parse_args()

    run_gene_dropout_analysis(
        base_dir=args.base_dir,
        tasks=args.tasks,
        n_repeats=args.n_repeats,
        dropout_fracs=args.dropout_fracs
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
