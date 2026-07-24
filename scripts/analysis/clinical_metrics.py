#!/usr/bin/env python3
"""
Clinical Metrics Analysis for Foundation Model Benchmark

Computes clinical ML metrics beyond standard AUC:
- AUPRC (Area Under Precision-Recall Curve)
- Sensitivity at 90% Specificity
- Specificity at 90% Sensitivity
- Confusion Matrix
- Cohen's Kappa
- Positive/Negative Predictive Value

Usage:
    python clinical_metrics.py --tasks binary 3-class
    python clinical_metrics.py --tasks binary

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
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, cohen_kappa_score, roc_curve,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, accuracy_score
)
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


def sensitivity_at_specificity(y_true, y_prob, target_specificity=0.9):
    """Calculate sensitivity at a given specificity level"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Specificity = 1 - FPR
    specificities = 1 - fpr

    # Find the threshold where specificity >= target
    idx = np.where(specificities >= target_specificity)[0]
    if len(idx) == 0:
        return 0.0, 0.0

    # Get the highest sensitivity at that specificity level
    best_idx = idx[np.argmax(tpr[idx])]
    return tpr[best_idx], thresholds[best_idx]


def specificity_at_sensitivity(y_true, y_prob, target_sensitivity=0.9):
    """Calculate specificity at a given sensitivity level"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Specificity = 1 - FPR
    specificities = 1 - fpr

    # Find the threshold where sensitivity >= target
    idx = np.where(tpr >= target_sensitivity)[0]
    if len(idx) == 0:
        return 0.0, 0.0

    # Get the highest specificity at that sensitivity level
    best_idx = idx[np.argmax(specificities[idx])]
    return specificities[best_idx], thresholds[best_idx]


def compute_clinical_metrics(X, y, n_splits=5):
    """Compute comprehensive clinical metrics"""
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

    # Get cross-validated predictions
    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=cv)
    y_prob = cross_val_predict(pipeline, X, y_encoded, cv=cv, method='predict_proba')

    metrics = {
        'n_samples': len(y),
        'n_classes': n_classes,
        'classes': le.classes_.tolist()
    }

    # Standard metrics
    metrics['accuracy'] = accuracy_score(y_encoded, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_encoded, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_encoded, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_encoded, y_pred, average=None).tolist()
    metrics['recall_per_class'] = recall_score(y_encoded, y_pred, average=None).tolist()
    metrics['f1_per_class'] = f1_score(y_encoded, y_pred, average=None).tolist()

    # Macro/weighted averages
    metrics['precision_macro'] = precision_score(y_encoded, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_encoded, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_encoded, y_pred, average='macro')

    # AUC metrics
    if n_classes == 2:
        # Binary classification
        metrics['auc_roc'] = roc_auc_score(y_encoded, y_prob[:, 1])

        # AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(y_encoded, y_prob[:, 1])
        metrics['auc_pr'] = auc(recall_curve, precision_curve)

        # Sensitivity at 90% specificity
        sens_at_spec, _ = sensitivity_at_specificity(y_encoded, y_prob[:, 1], 0.9)
        metrics['sensitivity_at_90_specificity'] = sens_at_spec

        # Specificity at 90% sensitivity
        spec_at_sens, _ = specificity_at_sensitivity(y_encoded, y_prob[:, 1], 0.9)
        metrics['specificity_at_90_sensitivity'] = spec_at_sens

        # PPV and NPV at default threshold (0.5)
        tn, fp, fn, tp = cm.ravel()
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    else:
        # Multi-class
        metrics['auc_roc_macro'] = roc_auc_score(y_encoded, y_prob, multi_class='ovr', average='macro')
        metrics['auc_roc_weighted'] = roc_auc_score(y_encoded, y_prob, multi_class='ovr', average='weighted')

        # Per-class AUPRC
        auprc_per_class = []
        for i in range(n_classes):
            y_binary = (y_encoded == i).astype(int)
            precision_curve, recall_curve, _ = precision_recall_curve(y_binary, y_prob[:, i])
            auprc_per_class.append(auc(recall_curve, precision_curve))
        metrics['auc_pr_per_class'] = auprc_per_class
        metrics['auc_pr_macro'] = np.mean(auprc_per_class)

    return metrics


def run_clinical_metrics_analysis(base_dir, tasks=None, n_splits=5, models=None):
    """Run clinical metrics analysis for all models"""

    output_dir = Path(base_dir) / "05_RESULTS/covid_severity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = ['binary', '3-class']

    if models is None:
        models = ['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer']

    logger.info("="*80)
    logger.info("CLINICAL METRICS ANALYSIS")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"CV Folds: {n_splits}")
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

                logger.info(f"  Samples: {len(y_valid)}")

                metrics = compute_clinical_metrics(X_valid, y_valid, n_splits=n_splits)
                metrics['model'] = model_name
                task_results[model_name] = metrics

                # Log key metrics
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
                logger.info(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")

                if task_name == 'binary':
                    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
                    logger.info(f"  Sens@90%Spec: {metrics['sensitivity_at_90_specificity']:.4f}")
                    logger.info(f"  Spec@90%Sens: {metrics['specificity_at_90_sensitivity']:.4f}")
                else:
                    logger.info(f"  AUC-ROC (macro): {metrics['auc_roc_macro']:.4f}")
                    logger.info(f"  AUC-PR (macro): {metrics['auc_pr_macro']:.4f}")

            except Exception as e:
                logger.error(f"  Error: {str(e)}")

        all_results[task_name] = task_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (full results)
    json_file = output_dir / f'clinical_metrics_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved: {json_file}")

    # CSV summary
    rows = []
    for task_name, task_results in all_results.items():
        for model_name, metrics in task_results.items():
            row = {
                'Task': task_name,
                'Model': model_name,
                'N_Samples': metrics['n_samples'],
                'Accuracy': metrics['accuracy'],
                'Balanced_Accuracy': metrics['balanced_accuracy'],
                'Cohen_Kappa': metrics['cohen_kappa'],
                'Precision_Macro': metrics['precision_macro'],
                'Recall_Macro': metrics['recall_macro'],
                'F1_Macro': metrics['f1_macro'],
            }

            if task_name == 'binary':
                row.update({
                    'AUC_ROC': metrics['auc_roc'],
                    'AUC_PR': metrics['auc_pr'],
                    'Sensitivity_at_90Spec': metrics['sensitivity_at_90_specificity'],
                    'Specificity_at_90Sens': metrics['specificity_at_90_sensitivity'],
                    'PPV': metrics['ppv'],
                    'NPV': metrics['npv'],
                    'Sensitivity': metrics['sensitivity'],
                    'Specificity': metrics['specificity'],
                })
            else:
                row.update({
                    'AUC_ROC_Macro': metrics['auc_roc_macro'],
                    'AUC_ROC_Weighted': metrics['auc_roc_weighted'],
                    'AUC_PR_Macro': metrics['auc_pr_macro'],
                })

            rows.append(row)

    df = pd.DataFrame(rows)
    csv_file = output_dir / f'clinical_metrics_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("CLINICAL METRICS SUMMARY")
    print(f"{'='*80}")

    for task_name in all_results.keys():
        print(f"\n{task_name}:")
        print("-" * 80)

        if task_name == 'binary':
            print(f"{'Model':<18} {'AUC-ROC':>10} {'AUC-PR':>10} {'Sens@90Sp':>10} {'Kappa':>10}")
            print("-" * 80)
            for model_name, metrics in all_results[task_name].items():
                print(f"{model_name:<18} {metrics['auc_roc']:>10.4f} {metrics['auc_pr']:>10.4f} "
                      f"{metrics['sensitivity_at_90_specificity']:>10.4f} {metrics['cohen_kappa']:>10.4f}")
        else:
            print(f"{'Model':<18} {'AUC-ROC':>10} {'AUC-PR':>10} {'Kappa':>10} {'BalAcc':>10}")
            print("-" * 80)
            for model_name, metrics in all_results[task_name].items():
                print(f"{model_name:<18} {metrics['auc_roc_macro']:>10.4f} {metrics['auc_pr_macro']:>10.4f} "
                      f"{metrics['cohen_kappa']:>10.4f} {metrics['balanced_accuracy']:>10.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Clinical Metrics Analysis')
    parser.add_argument('--base-dir', type=str,
                        default="/bigdata/godziklab/shared/Xinru/302006",
                        help='Base directory')
    parser.add_argument('--tasks', nargs='+', default=['binary', '3-class'],
                        choices=['binary', '3-class', '6-class'],
                        help='Tasks to analyze')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--models', nargs='+',
                        default=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        choices=['STATE', 'UCE', 'TranscriptFormer', 'scGPT', 'Geneformer'],
                        help='Models to analyze')
    args = parser.parse_args()

    run_clinical_metrics_analysis(
        base_dir=args.base_dir,
        tasks=args.tasks,
        n_splits=args.cv_folds,
        models=args.models
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
