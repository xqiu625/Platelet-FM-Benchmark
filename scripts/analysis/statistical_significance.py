#!/usr/bin/env python3
"""
Statistical Significance Analysis for COVID Severity Benchmark

Performs:
1. Bootstrap confidence intervals for AUC (n=1000)
2. DeLong test for AUC differences between models
3. Per-fold AUC values for violin plots

Usage:
    python statistical_significance.py --tasks binary 3-class
    python statistical_significance.py --tasks binary --n-bootstrap 500

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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from scipy import stats
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
                embedding_dir / "uce/covid_uce.h5ad",
            ],
            'obsm_keys': ['X_uce', 'X_uce_4layer'],
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


def compute_auc(y_true, y_prob, n_classes):
    """Compute AUC for binary or multi-class"""
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')


def bootstrap_auc(X, y, n_bootstrap=1000, random_state=42):
    """Compute bootstrap confidence interval for AUC"""
    np.random.seed(random_state)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    n_samples = len(y)

    # Fit model on full data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=2000, random_state=42, n_jobs=-1,
            class_weight='balanced', solver='saga'
        ))
    ])
    pipeline.fit(X, y_encoded)

    # Original AUC
    y_prob = pipeline.predict_proba(X)
    original_auc = compute_auc(y_encoded, y_prob, n_classes)

    # Bootstrap
    bootstrap_aucs = []
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y_encoded[indices]

        # Check if all classes present
        if len(np.unique(y_boot)) < n_classes:
            continue

        try:
            pipeline_boot = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    max_iter=1000, random_state=i, n_jobs=1,
                    class_weight='balanced', solver='saga'
                ))
            ])
            pipeline_boot.fit(X_boot, y_boot)
            y_prob_boot = pipeline_boot.predict_proba(X_boot)
            auc_boot = compute_auc(y_boot, y_prob_boot, n_classes)
            bootstrap_aucs.append(auc_boot)
        except:
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)

    return {
        'auc': original_auc,
        'ci_lower': np.percentile(bootstrap_aucs, 2.5),
        'ci_upper': np.percentile(bootstrap_aucs, 97.5),
        'std': np.std(bootstrap_aucs),
        'n_bootstrap': len(bootstrap_aucs)
    }


def get_fold_aucs(X, y, n_splits=5):
    """Get AUC for each CV fold (for violin plots)"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=2000, random_state=42, n_jobs=-1,
                class_weight='balanced', solver='saga'
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)

        try:
            auc = compute_auc(y_test, y_prob, n_classes)
            fold_aucs.append({'fold': fold_idx + 1, 'auc': auc})
        except:
            pass

    return fold_aucs


def delong_test(auc1, auc2, n1, n2):
    """
    Simplified DeLong test approximation for AUC comparison.
    Returns z-statistic and p-value.
    """
    # Variance approximation (Hanley & McNeil)
    q1 = auc1 / (2 - auc1)
    q2 = auc2 / (2 - auc2)

    se1 = np.sqrt((auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1**2) + (n1 - 1) * (q2 - auc1**2)) / n1)
    se2 = np.sqrt((auc2 * (1 - auc2) + (n2 - 1) * (q1 - auc2**2) + (n2 - 1) * (q2 - auc2**2)) / n2)

    se_diff = np.sqrt(se1**2 + se2**2)

    if se_diff == 0:
        return 0, 1.0

    z = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def run_statistical_analysis(base_dir, tasks=None, n_bootstrap=1000, n_splits=5, models=None):
    """Run full statistical analysis"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['binary', '3-class']

    if models is None:
        models = ['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer']

    logger.info("="*80)
    logger.info("STATISTICAL SIGNIFICANCE ANALYSIS")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Bootstrap iterations: {n_bootstrap}")
    logger.info(f"CV Folds: {n_splits}")
    logger.info("="*80)

    all_results = {}

    for task_name in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_name}")
        logger.info(f"{'='*60}")

        task_results = {
            'bootstrap_ci': {},
            'fold_aucs': {},
            'delong_tests': {}
        }

        model_aucs = {}

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

                logger.info(f"  Samples: {len(y_valid)}")

                # Bootstrap CI
                logger.info(f"  Computing bootstrap CI (n={n_bootstrap})...")
                bootstrap_result = bootstrap_auc(X_valid, y_valid, n_bootstrap=n_bootstrap)
                task_results['bootstrap_ci'][model_name] = bootstrap_result
                model_aucs[model_name] = (bootstrap_result['auc'], len(y_valid))

                logger.info(f"  AUC: {bootstrap_result['auc']:.4f} "
                           f"[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")

                # Fold AUCs
                logger.info(f"  Computing per-fold AUCs...")
                fold_aucs = get_fold_aucs(X_valid, y_valid, n_splits=n_splits)
                task_results['fold_aucs'][model_name] = fold_aucs

                fold_auc_values = [f['auc'] for f in fold_aucs]
                logger.info(f"  Fold AUCs: {[f'{x:.4f}' for x in fold_auc_values]}")

            except Exception as e:
                logger.error(f"  Error: {str(e)}")

        # DeLong tests (pairwise)
        logger.info(f"\n--- DeLong Tests ---")
        model_names = list(model_aucs.keys())

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                auc1, n1 = model_aucs[model1]
                auc2, n2 = model_aucs[model2]

                z, p = delong_test(auc1, auc2, n1, n2)

                comparison_key = f"{model1}_vs_{model2}"
                task_results['delong_tests'][comparison_key] = {
                    'model1': model1,
                    'model2': model2,
                    'auc1': auc1,
                    'auc2': auc2,
                    'auc_diff': auc1 - auc2,
                    'z_statistic': z,
                    'p_value': p,
                    'significant': p < 0.05
                }

                sig_str = "*" if p < 0.05 else ""
                logger.info(f"  {model1} vs {model2}: "
                           f"AUC diff = {auc1-auc2:+.4f}, p = {p:.4f} {sig_str}")

        all_results[task_name] = task_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_file = output_dir / f'statistical_significance_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_file}")

    # CSV summary - Bootstrap CI
    rows = []
    for task_name, task_results in all_results.items():
        for model_name, ci_result in task_results['bootstrap_ci'].items():
            rows.append({
                'Task': task_name,
                'Model': model_name,
                'AUC': ci_result['auc'],
                'CI_Lower': ci_result['ci_lower'],
                'CI_Upper': ci_result['ci_upper'],
                'CI_Width': ci_result['ci_upper'] - ci_result['ci_lower'],
                'Std': ci_result['std']
            })

    df_ci = pd.DataFrame(rows)
    csv_ci = output_dir / f'bootstrap_ci_{timestamp}.csv'
    df_ci.to_csv(csv_ci, index=False)
    logger.info(f"Saved: {csv_ci}")

    # CSV summary - Fold AUCs (for violin plots)
    rows_fold = []
    for task_name, task_results in all_results.items():
        for model_name, fold_aucs in task_results['fold_aucs'].items():
            for fold_result in fold_aucs:
                rows_fold.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Fold': fold_result['fold'],
                    'AUC': fold_result['auc']
                })

    df_fold = pd.DataFrame(rows_fold)
    csv_fold = output_dir / f'fold_aucs_{timestamp}.csv'
    df_fold.to_csv(csv_fold, index=False)
    logger.info(f"Saved: {csv_fold}")

    # CSV summary - DeLong tests
    rows_delong = []
    for task_name, task_results in all_results.items():
        for comp_name, delong_result in task_results['delong_tests'].items():
            rows_delong.append({
                'Task': task_name,
                'Comparison': comp_name,
                'Model1': delong_result['model1'],
                'Model2': delong_result['model2'],
                'AUC1': delong_result['auc1'],
                'AUC2': delong_result['auc2'],
                'AUC_Diff': delong_result['auc_diff'],
                'Z_Statistic': delong_result['z_statistic'],
                'P_Value': delong_result['p_value'],
                'Significant': delong_result['significant']
            })

    df_delong = pd.DataFrame(rows_delong)
    csv_delong = output_dir / f'delong_tests_{timestamp}.csv'
    df_delong.to_csv(csv_delong, index=False)
    logger.info(f"Saved: {csv_delong}")

    # Print summary
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print(f"{'='*80}")

    for task_name in all_results.keys():
        print(f"\n{task_name}:")
        print("-" * 60)
        print(f"{'Model':<20} {'AUC':>8} {'95% CI':>20}")
        print("-" * 60)

        for model_name, ci in all_results[task_name]['bootstrap_ci'].items():
            ci_str = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
            print(f"{model_name:<20} {ci['auc']:>8.4f} {ci_str:>20}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Analysis')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory')
    parser.add_argument('--tasks', nargs='+', default=['binary', '3-class'],
                        choices=['binary', '3-class', '6-class'],
                        help='Tasks to analyze')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--models', nargs='+',
                        default=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        choices=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        help='Models to analyze')
    args = parser.parse_args()

    run_statistical_analysis(
        base_dir=args.base_dir,
        tasks=args.tasks,
        n_bootstrap=args.n_bootstrap,
        n_splits=args.cv_folds,
        models=args.models
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
