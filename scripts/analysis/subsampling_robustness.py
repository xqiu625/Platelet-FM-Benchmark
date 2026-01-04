#!/usr/bin/env python3
"""
Subsampling Robustness Analysis for Foundation Model Benchmark

Tests how model performance degrades with reduced training data:
- 100%, 50%, 20%, 10%, 5% of cells
- Measures AUC at each subsample level
- Generates robustness curves

Usage:
    python subsampling_robustness.py --tasks binary 3-class
    python subsampling_robustness.py --tasks binary --n-repeats 5

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

from sklearn.model_selection import StratifiedKFold, train_test_split
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


def evaluate_at_subsample(X, y, subsample_frac, n_repeats=5, test_size=0.2, random_state=42):
    """Evaluate model at a specific subsample fraction"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    aucs = []
    balanced_accs = []

    for repeat in range(n_repeats):
        seed = random_state + repeat

        # Split into train/test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
        )

        # Subsample training data
        if subsample_frac < 1.0:
            n_subsample = max(int(len(X_train_full) * subsample_frac), n_classes * 2)
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_train_full, y_train_full,
                    train_size=n_subsample,
                    random_state=seed,
                    stratify=y_train_full
                )
            except:
                # If stratified split fails, use random
                indices = np.random.RandomState(seed).choice(
                    len(X_train_full), n_subsample, replace=False
                )
                X_train = X_train_full[indices]
                y_train = y_train_full[indices]
        else:
            X_train = X_train_full
            y_train = y_train_full

        # Check if all classes present
        if len(np.unique(y_train)) < n_classes:
            continue

        # Train and evaluate
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    max_iter=2000, random_state=seed, n_jobs=-1,
                    class_weight='balanced', solver='saga'
                ))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)

            auc = compute_auc(y_test, y_prob, n_classes)
            bal_acc = balanced_accuracy_score(y_test, y_pred)

            aucs.append(auc)
            balanced_accs.append(bal_acc)

        except Exception as e:
            logger.warning(f"  Repeat {repeat+1} failed: {e}")

    if not aucs:
        return None

    return {
        'subsample_frac': subsample_frac,
        'n_train_samples': int(len(X) * (1 - test_size) * subsample_frac),
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'auc_min': np.min(aucs),
        'auc_max': np.max(aucs),
        'balanced_acc_mean': np.mean(balanced_accs),
        'balanced_acc_std': np.std(balanced_accs),
        'n_repeats': len(aucs)
    }


def run_subsampling_analysis(base_dir, tasks=None, n_repeats=5, subsample_fracs=None):
    """Run subsampling robustness analysis"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['binary', '3-class']

    if subsample_fracs is None:
        subsample_fracs = [1.0, 0.5, 0.2, 0.1, 0.05]

    models = ['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer']

    logger.info("="*80)
    logger.info("SUBSAMPLING ROBUSTNESS ANALYSIS")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Subsample fractions: {subsample_fracs}")
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

                logger.info(f"  Total samples: {len(y_valid)}")

                model_results = []

                for frac in subsample_fracs:
                    logger.info(f"  Subsample {frac*100:.0f}%...")

                    result = evaluate_at_subsample(
                        X_valid, y_valid,
                        subsample_frac=frac,
                        n_repeats=n_repeats
                    )

                    if result:
                        result['model'] = model_name
                        model_results.append(result)
                        logger.info(f"    AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f} "
                                   f"(n_train={result['n_train_samples']})")

                task_results[model_name] = model_results

            except Exception as e:
                logger.error(f"  Error: {str(e)}")

        all_results[task_name] = task_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_file = output_dir / f'subsampling_robustness_{timestamp}.json'
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
                    'Subsample_Frac': result['subsample_frac'],
                    'N_Train_Samples': result['n_train_samples'],
                    'AUC_Mean': result['auc_mean'],
                    'AUC_Std': result['auc_std'],
                    'AUC_Min': result['auc_min'],
                    'AUC_Max': result['auc_max'],
                    'BalancedAcc_Mean': result['balanced_acc_mean'],
                    'BalancedAcc_Std': result['balanced_acc_std'],
                    'N_Repeats': result['n_repeats']
                })

    df = pd.DataFrame(rows)
    csv_file = output_dir / f'subsampling_robustness_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUBSAMPLING ROBUSTNESS SUMMARY")
    print(f"{'='*80}")

    for task_name in all_results.keys():
        print(f"\n{task_name}:")
        print("-" * 70)
        print(f"{'Model':<18} {'100%':>10} {'50%':>10} {'20%':>10} {'10%':>10} {'5%':>10}")
        print("-" * 70)

        for model_name in models:
            if model_name in all_results[task_name]:
                results = all_results[task_name][model_name]
                values = {r['subsample_frac']: r['auc_mean'] for r in results}

                row = f"{model_name:<18}"
                for frac in [1.0, 0.5, 0.2, 0.1, 0.05]:
                    if frac in values:
                        row += f" {values[frac]:>9.4f}"
                    else:
                        row += f" {'N/A':>9}"
                print(row)

    print(f"\n{'='*80}")
    print("Interpretation:")
    print("  - Flat curve = robust to data reduction")
    print("  - Steep drop = sensitive to training size")
    print("  - Foundation models should maintain performance better than baselines")
    print(f"{'='*80}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Subsampling Robustness Analysis')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory')
    parser.add_argument('--tasks', nargs='+', default=['binary', '3-class'],
                        choices=['binary', '3-class', '6-class'],
                        help='Tasks to analyze')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Number of repeats per subsample level')
    parser.add_argument('--subsample-fracs', nargs='+', type=float,
                        default=[1.0, 0.5, 0.2, 0.1, 0.05],
                        help='Subsample fractions to test')
    args = parser.parse_args()

    run_subsampling_analysis(
        base_dir=args.base_dir,
        tasks=args.tasks,
        n_repeats=args.n_repeats,
        subsample_fracs=args.subsample_fracs
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
