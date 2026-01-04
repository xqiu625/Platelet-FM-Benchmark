#!/usr/bin/env python3
"""
Embedding Ablation Analysis for Foundation Model Benchmark

Tests how embedding modifications affect performance:
1. Dimension reduction (128, 256, 512 dims via PCA)
2. Remove top PCA components (1, 2, 5, 10)
3. Normalize vs unnormalized embeddings
4. Random dimension selection

Usage:
    python embedding_ablations.py --tasks binary 3-class
    python embedding_ablations.py --tasks binary --ablations dim_reduction normalize

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
from sklearn.decomposition import PCA
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
            'dims': 1280,
        },
        'scGPT': {
            'files': [
                embedding_dir / "scgpt/covid_scgpt.h5ad",
                embedding_dir / "scgpt/platelet_scgpt_combined.h5ad",
            ],
            'obsm_keys': ['X_scGPT', 'X_scgpt'],
            'dims': 512,
        },
        'TranscriptFormer': {
            'files': [
                embedding_dir / "transcriptformer/covid_transcriptformer.h5ad",
            ],
            'obsm_keys': ['X_transcriptformer', 'X_TranscriptFormer'],
            'dims': 2048,
        },
        'Geneformer': {
            'files': [
                embedding_dir / "geneformer/covid_geneformer.h5ad",
            ],
            'obsm_keys': ['X_geneformer', 'X_Geneformer'],
            'dims': 1152,
        },
        'STATE': {
            'files': [
                embedding_dir / "state/covid_state.h5ad",
            ],
            'obsm_keys': ['X_emb', 'X_state'],
            'dims': 2058,
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


def compute_auc(y_true, y_prob, n_classes):
    """Compute AUC for binary or multi-class"""
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')


def evaluate_embeddings(X, y, n_splits=5):
    """Evaluate embeddings with cross-validation"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=2000, random_state=42, n_jobs=-1,
            class_weight='balanced', solver='saga'
        ))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        y_pred = cross_val_predict(pipeline, X, y_encoded, cv=cv)
        y_prob = cross_val_predict(pipeline, X, y_encoded, cv=cv, method='predict_proba')

        auc = compute_auc(y_encoded, y_prob, n_classes)
        bal_acc = balanced_accuracy_score(y_encoded, y_pred)

        return {'auc': auc, 'balanced_accuracy': bal_acc}
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return None


def ablation_dim_reduction(X, y, target_dims=[64, 128, 256, 512]):
    """Reduce embedding dimensions using PCA"""
    results = []

    # Original
    logger.info(f"  Original ({X.shape[1]} dims)...")
    original_result = evaluate_embeddings(X, y)
    if original_result:
        results.append({
            'ablation': 'dim_reduction',
            'setting': 'original',
            'n_dims': X.shape[1],
            **original_result
        })

    # Reduced dimensions
    for n_dims in target_dims:
        if n_dims >= X.shape[1]:
            continue

        logger.info(f"  PCA to {n_dims} dims...")
        pca = PCA(n_components=n_dims, random_state=42)
        X_reduced = pca.fit_transform(StandardScaler().fit_transform(X))

        result = evaluate_embeddings(X_reduced, y)
        if result:
            results.append({
                'ablation': 'dim_reduction',
                'setting': f'pca_{n_dims}',
                'n_dims': n_dims,
                'variance_explained': pca.explained_variance_ratio_.sum(),
                **result
            })

    return results


def ablation_remove_top_pca(X, y, n_components_to_remove=[1, 2, 5, 10]):
    """Remove top PCA components (which capture most variance)"""
    results = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Original (all components)
    logger.info(f"  All PCA components...")
    original_result = evaluate_embeddings(X_pca, y)
    if original_result:
        results.append({
            'ablation': 'remove_top_pca',
            'setting': 'all_components',
            'n_removed': 0,
            **original_result
        })

    # Remove top N components
    for n_remove in n_components_to_remove:
        if n_remove >= X_pca.shape[1]:
            continue

        logger.info(f"  Remove top {n_remove} components...")
        X_ablated = X_pca[:, n_remove:]  # Remove first N columns

        result = evaluate_embeddings(X_ablated, y)
        if result:
            variance_removed = pca.explained_variance_ratio_[:n_remove].sum()
            results.append({
                'ablation': 'remove_top_pca',
                'setting': f'remove_top_{n_remove}',
                'n_removed': n_remove,
                'variance_removed': variance_removed,
                'n_dims_remaining': X_ablated.shape[1],
                **result
            })

    return results


def ablation_normalization(X, y):
    """Compare normalized vs unnormalized embeddings"""
    results = []

    # Unnormalized (raw)
    logger.info(f"  Raw (unnormalized)...")
    raw_result = evaluate_embeddings(X, y)
    if raw_result:
        results.append({
            'ablation': 'normalization',
            'setting': 'raw',
            **raw_result
        })

    # L2 normalized
    logger.info(f"  L2 normalized...")
    X_l2 = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    l2_result = evaluate_embeddings(X_l2, y)
    if l2_result:
        results.append({
            'ablation': 'normalization',
            'setting': 'l2_norm',
            **l2_result
        })

    # Z-score normalized
    logger.info(f"  Z-score normalized...")
    X_zscore = StandardScaler().fit_transform(X)
    zscore_result = evaluate_embeddings(X_zscore, y)
    if zscore_result:
        results.append({
            'ablation': 'normalization',
            'setting': 'zscore',
            **zscore_result
        })

    return results


def ablation_random_dims(X, y, n_dims_list=[64, 128, 256], n_repeats=3):
    """Select random dimensions (baseline comparison)"""
    results = []

    for n_dims in n_dims_list:
        if n_dims >= X.shape[1]:
            continue

        logger.info(f"  Random {n_dims} dims ({n_repeats} repeats)...")
        aucs = []
        bal_accs = []

        for repeat in range(n_repeats):
            np.random.seed(42 + repeat)
            random_indices = np.random.choice(X.shape[1], n_dims, replace=False)
            X_random = X[:, random_indices]

            result = evaluate_embeddings(X_random, y)
            if result:
                aucs.append(result['auc'])
                bal_accs.append(result['balanced_accuracy'])

        if aucs:
            results.append({
                'ablation': 'random_dims',
                'setting': f'random_{n_dims}',
                'n_dims': n_dims,
                'auc': np.mean(aucs),
                'auc_std': np.std(aucs),
                'balanced_accuracy': np.mean(bal_accs),
                'balanced_accuracy_std': np.std(bal_accs),
            })

    return results


def run_embedding_ablations(base_dir, tasks=None, ablations=None, models=None):
    """Run embedding ablation analysis"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['binary', '3-class']

    if ablations is None:
        ablations = ['dim_reduction', 'remove_top_pca', 'normalization', 'random_dims']

    if models is None:
        models = ['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer']

    logger.info("="*80)
    logger.info("EMBEDDING ABLATION ANALYSIS")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Ablations: {ablations}")
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

                logger.info(f"  Samples: {len(y_valid)}, Dims: {X_valid.shape[1]}")

                model_results = []

                if 'dim_reduction' in ablations:
                    logger.info(f"  [Dimension Reduction]")
                    model_results.extend(ablation_dim_reduction(X_valid, y_valid))

                if 'remove_top_pca' in ablations:
                    logger.info(f"  [Remove Top PCA]")
                    model_results.extend(ablation_remove_top_pca(X_valid, y_valid))

                if 'normalization' in ablations:
                    logger.info(f"  [Normalization]")
                    model_results.extend(ablation_normalization(X_valid, y_valid))

                if 'random_dims' in ablations:
                    logger.info(f"  [Random Dimensions]")
                    model_results.extend(ablation_random_dims(X_valid, y_valid))

                # Add model info
                for r in model_results:
                    r['model'] = model_name
                    r['original_dims'] = X_valid.shape[1]

                task_results[model_name] = model_results

            except Exception as e:
                logger.error(f"  Error: {str(e)}")

        all_results[task_name] = task_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_file = output_dir / f'embedding_ablations_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_file}")

    # CSV
    rows = []
    for task_name, task_results in all_results.items():
        for model_name, model_results in task_results.items():
            for result in model_results:
                row = {
                    'Task': task_name,
                    'Model': model_name,
                    'Ablation': result.get('ablation'),
                    'Setting': result.get('setting'),
                    'Original_Dims': result.get('original_dims'),
                    'N_Dims': result.get('n_dims'),
                    'AUC': result.get('auc'),
                    'AUC_Std': result.get('auc_std'),
                    'Balanced_Accuracy': result.get('balanced_accuracy'),
                    'Variance_Explained': result.get('variance_explained'),
                    'Variance_Removed': result.get('variance_removed'),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_file = output_dir / f'embedding_ablations_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("EMBEDDING ABLATION SUMMARY")
    print(f"{'='*80}")

    for task_name in all_results.keys():
        print(f"\n{task_name}:")

        for ablation_type in ablations:
            print(f"\n  [{ablation_type}]")
            print(f"  {'Model':<18} {'Setting':<20} {'AUC':>10}")
            print(f"  {'-'*50}")

            for model_name in models:
                if model_name in all_results[task_name]:
                    for result in all_results[task_name][model_name]:
                        if result.get('ablation') == ablation_type:
                            auc_str = f"{result['auc']:.4f}" if result.get('auc') else 'N/A'
                            print(f"  {model_name:<18} {result.get('setting', 'N/A'):<20} {auc_str:>10}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Embedding Ablation Analysis')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory')
    parser.add_argument('--tasks', nargs='+', default=['binary', '3-class'],
                        choices=['binary', '3-class', '6-class'],
                        help='Tasks to analyze')
    parser.add_argument('--ablations', nargs='+',
                        default=['dim_reduction', 'remove_top_pca', 'normalization', 'random_dims'],
                        choices=['dim_reduction', 'remove_top_pca', 'normalization', 'random_dims'],
                        help='Ablation types to run')
    parser.add_argument('--models', nargs='+',
                        default=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        choices=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        help='Models to analyze')
    args = parser.parse_args()

    run_embedding_ablations(
        base_dir=args.base_dir,
        tasks=args.tasks,
        ablations=args.ablations,
        models=args.models
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
