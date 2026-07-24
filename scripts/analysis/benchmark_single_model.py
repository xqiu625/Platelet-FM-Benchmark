#!/usr/bin/env python3
"""
COVID Severity Prediction Benchmark - Single Model Version

Run benchmark for ONE model at a time to fit within 2-hour GPU limits.

Usage:
    python benchmark_single_model.py --model UCE
    python benchmark_single_model.py --model scGPT
    python benchmark_single_model.py --model TranscriptFormer
    python benchmark_single_model.py --model Geneformer

After all 4 runs complete, use merge_benchmark_results.py to combine.

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
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
            'dims': 1280
        },
        'scGPT': {
            'files': [
                embedding_dir / "scgpt/covid_scgpt.h5ad",
                embedding_dir / "scgpt/platelet_scgpt_combined.h5ad",
            ],
            'obsm_keys': ['X_scGPT', 'X_scgpt'],
            'dims': 512
        },
        'TranscriptFormer': {
            'files': [
                embedding_dir / "transcriptformer/covid_transcriptformer.h5ad",
            ],
            'obsm_keys': ['X_transcriptformer', 'X_TranscriptFormer'],
            'dims': 2048
        },
        'Geneformer': {
            'files': [
                embedding_dir / "geneformer/covid_geneformer.h5ad",
            ],
            'obsm_keys': ['X_geneformer', 'X_Geneformer'],
            'dims': 1152
        }
    }

    return configs.get(model_name)


def load_single_model(model_name, base_dir):
    """Load embeddings and labels for a single model"""
    config = get_embedding_config(model_name, base_dir)

    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Find the file
    file_path = None
    for fp in config['files']:
        if fp.exists():
            file_path = fp
            break

    if file_path is None:
        raise FileNotFoundError(f"No file found for {model_name}. Tried: {config['files']}")

    logger.info(f"Loading {model_name} from {file_path}")
    adata = sc.read_h5ad(file_path)

    # For scGPT combined file, filter to COVID only
    if model_name == 'scGPT' and 'platelet_scgpt_combined' in str(file_path):
        if 'dataset' in adata.obs.columns:
            covid_mask = adata.obs['dataset'] == 'covid'
            if covid_mask.sum() > 0:
                adata = adata[covid_mask].copy()
                logger.info(f"  Filtered by dataset==covid: {adata.n_obs} cells")

        if 'covid_severity' in adata.obs.columns:
            valid_severities = ['healthy', 'mild', 'moderate', 'severe', 'fatal', 'recovered']
            severity_labels = np.array([str(x) if pd.notna(x) else 'unknown' for x in adata.obs['covid_severity'].values])
            valid_mask = np.isin(severity_labels, valid_severities)
            if valid_mask.sum() < len(adata):
                adata = adata[valid_mask].copy()
                logger.info(f"  Filtered by valid severity: {adata.n_obs} cells")

    # Get embeddings
    obsm_key = None
    for key in config['obsm_keys']:
        if key in adata.obsm:
            obsm_key = key
            break

    if obsm_key is None:
        possible_keys = [k for k in adata.obsm.keys() if model_name.lower() in k.lower()]
        if possible_keys:
            obsm_key = possible_keys[0]
        else:
            raise KeyError(f"Embedding key not found. Available: {list(adata.obsm.keys())}")

    embeddings = adata.obsm[obsm_key]
    if hasattr(embeddings, 'toarray'):
        embeddings = embeddings.toarray()
    embeddings = np.array(embeddings)

    # Get labels
    labels = None
    label_cols = ['covid_severity', 'severity', 'Category', 'three_group']
    for col in label_cols:
        if col in adata.obs.columns:
            raw_labels = adata.obs[col].values
            labels = np.array([str(x) if pd.notna(x) else 'unknown' for x in raw_labels])
            logger.info(f"  Using label column: {col}")
            break

    if labels is None:
        raise ValueError(f"No severity label column found. Available: {list(adata.obs.columns)}")

    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return embeddings, labels


def prepare_labels(task_name, labels):
    """Prepare labels for a specific classification task"""
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


def evaluate_classifier(X, y, classifier, n_splits=5):
    """Evaluate a classifier using cross-validation"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=cv)

    try:
        y_prob = cross_val_predict(pipeline, X, y_encoded, cv=cv, method='predict_proba')
        if n_classes == 2:
            auc_macro = roc_auc_score(y_encoded, y_prob[:, 1])
            auc_weighted = auc_macro
        else:
            auc_macro = roc_auc_score(y_encoded, y_prob, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_encoded, y_prob, multi_class='ovr', average='weighted')
    except:
        auc_macro = None
        auc_weighted = None

    metrics = {
        'accuracy': accuracy_score(y_encoded, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_encoded, y_pred),
        'f1_macro': f1_score(y_encoded, y_pred, average='macro'),
        'f1_weighted': f1_score(y_encoded, y_pred, average='weighted'),
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'n_samples': len(y),
        'n_classes': n_classes,
        'classes': le.classes_.tolist(),
        'confusion_matrix': confusion_matrix(y_encoded, y_pred).tolist()
    }

    return metrics


def get_classifiers(quick=False):
    """Get classifier dictionary"""
    classifiers = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000, random_state=42, n_jobs=-1, class_weight='balanced', solver='saga'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
    }

    if not quick:
        classifiers.update({
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1
            ),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        })

    return classifiers


def run_single_model_benchmark(model_name, base_dir, quick=False, tasks=None, n_splits=5):
    """Run benchmark for a single model"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['6-class', '3-class', 'binary']

    # Load embeddings
    X, labels = load_single_model(model_name, base_dir)

    # Get classifiers
    classifiers = get_classifiers(quick=quick)

    logger.info("="*80)
    logger.info(f"BENCHMARK: {model_name}")
    logger.info(f"Quick mode: {quick}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"CV Folds: {n_splits}")
    logger.info("="*80)

    results = {}

    for task_name in tasks:
        logger.info(f"\n--- Task: {task_name} ---")

        y, valid_mask = prepare_labels(task_name, labels)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        logger.info(f"Samples: {len(y_valid)}, Classes: {np.unique(y_valid)}")

        results[task_name] = {}

        for clf_name, clf in classifiers.items():
            try:
                clf_clone = clone(clf)
                metrics = evaluate_classifier(X_valid, y_valid, clf_clone, n_splits=n_splits)
                results[task_name][clf_name] = metrics

                auc_str = f"{metrics['auc_macro']:.3f}" if metrics['auc_macro'] is not None else 'N/A'
                logger.info(f"  {clf_name}: Acc={metrics['accuracy']:.3f}, "
                           f"BalAcc={metrics['balanced_accuracy']:.3f}, AUC={auc_str}")

            except Exception as e:
                logger.error(f"  {clf_name}: Failed - {str(e)}")
                results[task_name][clf_name] = {'error': str(e)}

    # Save results for this model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON results
    json_file = output_dir / f'benchmark_{model_name}_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({model_name: results}, f, indent=2, default=str)
    logger.info(f"Saved: {json_file}")

    # CSV summary
    rows = []
    for task_name, task_results in results.items():
        for clf_name, metrics in task_results.items():
            if 'error' not in metrics:
                rows.append({
                    'Task': task_name,
                    'Embedding': model_name,
                    'Classifier': clf_name,
                    'Accuracy': metrics['accuracy'],
                    'Balanced_Accuracy': metrics['balanced_accuracy'],
                    'F1_Macro': metrics['f1_macro'],
                    'F1_Weighted': metrics['f1_weighted'],
                    'AUC_Macro': metrics['auc_macro'],
                    'AUC_Weighted': metrics['auc_weighted'],
                    'N_Samples': metrics['n_samples'],
                    'N_Classes': metrics['n_classes']
                })

    df = pd.DataFrame(rows)
    csv_file = output_dir / f'benchmark_{model_name}_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved: {csv_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")

    for task in df['Task'].unique():
        task_df = df[df['Task'] == task]
        best_idx = task_df['Balanced_Accuracy'].idxmax()
        best = task_df.loc[best_idx]
        print(f"\n{task}: Best = {best['Classifier']}")
        print(f"  Balanced Acc: {best['Balanced_Accuracy']:.4f}, AUC: {best['AUC_Macro']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='COVID Severity Benchmark - Single Model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['UCE', 'scGPT', 'TranscriptFormer', 'Geneformer'],
                        help='Model to benchmark')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory for the project')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: only LogisticRegression and RandomForest')
    parser.add_argument('--tasks', nargs='+', default=None,
                        choices=['6-class', '3-class', 'binary'],
                        help='Classification tasks to run')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    run_single_model_benchmark(
        model_name=args.model,
        base_dir=args.base_dir,
        quick=args.quick,
        tasks=args.tasks,
        n_splits=args.cv_folds
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
