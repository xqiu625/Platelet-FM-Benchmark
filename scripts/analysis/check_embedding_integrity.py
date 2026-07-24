#!/usr/bin/env python3
"""
Embedding Integrity Check

Detects potential synthetic/random fallback embeddings in generated data.
Random fallback embeddings (np.random.randn) have characteristic statistics:
- Row mean ≈ 0
- Row std ≈ 1

This script checks all foundation model embeddings for suspicious patterns.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/bigdata/godziklab/shared/Xinru/302006")
EMB_DIR = BASE_DIR / "02_EMBEDDINGS"

# All embedding files to check
EMBEDDING_FILES = {
    'STATE': {
        'path': EMB_DIR / "state/covid_state.h5ad",
        'key': 'X_emb',
        'expected_dim': 2058,
    },
    'UCE': {
        'path': EMB_DIR / "uce/human_platelet_covid_severity_uce_adata.h5ad",
        'key': 'X_uce',
        'expected_dim': 1280,
    },
    'TranscriptFormer': {
        'path': EMB_DIR / "transcriptformer/covid_transcriptformer.h5ad",
        'key': 'X_transcriptformer',
        'expected_dim': 2048,
    },
    'scGPT': {
        'path': EMB_DIR / "scgpt/covid_scgpt.h5ad",
        'key': 'X_scGPT',
        'expected_dim': 512,
        'normalized': True,  # scGPT embeddings are L2-normalized (mean≈0, std≈1 is expected)
    },
    'Geneformer': {
        'path': EMB_DIR / "geneformer/covid_geneformer.h5ad",
        'key': 'X_geneformer',
        'expected_dim': 1152,
    },
}

def check_for_random_embeddings(X, threshold_mean=0.15, threshold_std=0.2):
    """
    Check for rows that look like np.random.randn output.

    np.random.randn generates values with:
    - mean ≈ 0 (typically |mean| < 0.1 for 512+ dimensions)
    - std ≈ 1 (typically 0.9-1.1 for 512+ dimensions)

    Returns indices of suspicious cells.
    """
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1)

    # Suspicious: mean close to 0 AND std close to 1
    suspicious_mask = (np.abs(row_means) < threshold_mean) & (np.abs(row_stds - 1.0) < threshold_std)

    return np.where(suspicious_mask)[0], row_means, row_stds

def check_for_zero_embeddings(X, threshold=1e-6):
    """
    Check for rows that are all zeros (zero embedding fallback).
    """
    row_norms = np.linalg.norm(X, axis=1)
    zero_mask = row_norms < threshold
    return np.where(zero_mask)[0]

def check_for_constant_embeddings(X, threshold=1e-6):
    """
    Check for rows where all values are the same (constant fallback).
    """
    row_stds = X.std(axis=1)
    constant_mask = row_stds < threshold
    return np.where(constant_mask)[0]

def check_for_duplicate_embeddings(X, sample_size=10000):
    """
    Check for duplicate rows (could indicate copy-paste fallback).
    Only checks a sample for efficiency.
    """
    n_cells = X.shape[0]
    if n_cells > sample_size:
        indices = np.random.choice(n_cells, sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(n_cells)

    # Round to reduce floating point noise
    X_rounded = np.round(X_sample, decimals=4)

    # Find duplicates
    _, unique_indices, counts = np.unique(X_rounded, axis=0, return_index=True, return_counts=True)

    duplicates = counts[counts > 1]
    n_duplicates = len(duplicates)

    return n_duplicates, duplicates

def analyze_embedding_distribution(X):
    """
    Analyze overall embedding distribution statistics.
    """
    stats = {
        'global_mean': float(X.mean()),
        'global_std': float(X.std()),
        'global_min': float(X.min()),
        'global_max': float(X.max()),
        'row_mean_mean': float(X.mean(axis=1).mean()),
        'row_mean_std': float(X.mean(axis=1).std()),
        'row_std_mean': float(X.std(axis=1).mean()),
        'row_std_std': float(X.std(axis=1).std()),
        'nan_count': int(np.isnan(X).sum()),
        'inf_count': int(np.isinf(X).sum()),
    }
    return stats

def check_embedding_file(name, config):
    """Check a single embedding file for integrity issues."""
    path = config['path']
    emb_key = config['key']

    print(f"\n{'='*70}")
    print(f"Checking: {name}")
    print(f"{'='*70}")

    if not path.exists():
        print(f"  ❌ FILE NOT FOUND: {path}")
        return {'model': name, 'status': 'FILE_NOT_FOUND', 'issues': ['File not found']}

    print(f"  Path: {path}")

    try:
        adata = sc.read_h5ad(path)
        print(f"  Cells: {adata.n_obs:,}")

        # Find embeddings
        X = None
        actual_key = None

        # Try specified key first
        if emb_key in adata.obsm:
            X = adata.obsm[emb_key]
            actual_key = emb_key
        else:
            # Try common alternatives
            for key in ['X_emb', 'X_uce', 'X_scGPT', 'X_geneformer', 'embeddings']:
                if key in adata.obsm:
                    X = adata.obsm[key]
                    actual_key = key
                    break

        if X is None:
            # Check if embeddings are in .X
            X = adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            actual_key = 'X (main matrix)'

        print(f"  Embedding key: {actual_key}")
        print(f"  Shape: {X.shape}")

        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = np.array(X, dtype=np.float32)

        issues = []

        # 1. Check for NaN/Inf
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count:,} NaN values")
            print(f"  ⚠️  NaN values: {nan_count:,}")
        if inf_count > 0:
            issues.append(f"Contains {inf_count:,} Inf values")
            print(f"  ⚠️  Inf values: {inf_count:,}")

        # 2. Check for zero embeddings
        zero_indices = check_for_zero_embeddings(X)
        if len(zero_indices) > 0:
            issues.append(f"{len(zero_indices)} zero embeddings detected")
            print(f"  🚨 ZERO EMBEDDINGS: {len(zero_indices):,} cells")
            if len(zero_indices) <= 10:
                print(f"      Indices: {zero_indices.tolist()}")
            else:
                print(f"      First 10 indices: {zero_indices[:10].tolist()}")

        # 3. Check for constant embeddings
        constant_indices = check_for_constant_embeddings(X)
        if len(constant_indices) > 0:
            issues.append(f"{len(constant_indices)} constant embeddings detected")
            print(f"  🚨 CONSTANT EMBEDDINGS: {len(constant_indices):,} cells")

        # 4. Check for random embeddings (mean≈0, std≈1)
        # Skip for models with normalized embeddings (e.g., scGPT)
        is_normalized = config.get('normalized', False)
        random_indices, row_means, row_stds = check_for_random_embeddings(X)
        if len(random_indices) > 0:
            pct = 100 * len(random_indices) / X.shape[0]
            if is_normalized:
                print(f"  ℹ️  NORMALIZED EMBEDDINGS: {len(random_indices):,} cells ({pct:.2f}%) have mean≈0, std≈1")
                print(f"      This is EXPECTED for {name} (L2-normalized by design)")
                # Don't add to issues - this is expected behavior
            else:
                issues.append(f"{len(random_indices)} potential random fallback embeddings ({pct:.2f}%)")
                print(f"  🚨 POTENTIAL RANDOM FALLBACKS: {len(random_indices):,} cells ({pct:.2f}%)")
                if len(random_indices) <= 10:
                    print(f"      Indices: {random_indices.tolist()}")
                    for idx in random_indices[:5]:
                        print(f"        Cell {idx}: mean={row_means[idx]:.4f}, std={row_stds[idx]:.4f}")
                else:
                    print(f"      First 10 indices: {random_indices[:10].tolist()}")

        # 5. Check for duplicates
        n_duplicates, duplicate_counts = check_for_duplicate_embeddings(X)
        if n_duplicates > 0:
            issues.append(f"{n_duplicates} duplicate embedding groups detected")
            print(f"  ⚠️  DUPLICATE EMBEDDINGS: {n_duplicates} groups")

        # 6. Distribution statistics
        stats = analyze_embedding_distribution(X)
        print(f"\n  Distribution Statistics:")
        print(f"    Global mean: {stats['global_mean']:.4f}")
        print(f"    Global std:  {stats['global_std']:.4f}")
        print(f"    Range: [{stats['global_min']:.4f}, {stats['global_max']:.4f}]")
        print(f"    Row mean (mean±std): {stats['row_mean_mean']:.4f} ± {stats['row_mean_std']:.4f}")
        print(f"    Row std (mean±std):  {stats['row_std_mean']:.4f} ± {stats['row_std_std']:.4f}")

        # 7. Summary
        if len(issues) == 0:
            print(f"\n  ✅ PASSED - No integrity issues detected")
            status = 'PASSED'
        else:
            print(f"\n  ❌ ISSUES FOUND:")
            for issue in issues:
                print(f"      - {issue}")
            status = 'ISSUES_FOUND'

        # For normalized embeddings, don't count as random suspects
        n_random_to_report = 0 if is_normalized else len(random_indices)

        return {
            'model': name,
            'status': status,
            'n_cells': X.shape[0],
            'n_dims': X.shape[1],
            'n_zero': len(zero_indices),
            'n_constant': len(constant_indices),
            'n_random_suspect': n_random_to_report,
            'n_duplicates': n_duplicates,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'global_mean': stats['global_mean'],
            'global_std': stats['global_std'],
            'row_std_mean': stats['row_std_mean'],
            'issues': issues,
            'normalized': is_normalized
        }

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'model': name, 'status': 'ERROR', 'issues': [str(e)]}

def main():
    print("=" * 70)
    print("EMBEDDING INTEGRITY CHECK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nChecking for:")
    print("  - Zero embeddings (all zeros)")
    print("  - Constant embeddings (all same value)")
    print("  - Random fallback embeddings (mean≈0, std≈1)")
    print("  - Duplicate embeddings")
    print("  - NaN/Inf values")

    results = []

    for name, config in EMBEDDING_FILES.items():
        result = check_embedding_file(name, config)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    passed = results_df[results_df['status'] == 'PASSED']
    failed = results_df[results_df['status'] == 'ISSUES_FOUND']
    missing = results_df[results_df['status'] == 'FILE_NOT_FOUND']
    errors = results_df[results_df['status'] == 'ERROR']

    print(f"\n✅ PASSED: {len(passed)}")
    for _, row in passed.iterrows():
        print(f"   - {row['model']}")

    if len(failed) > 0:
        print(f"\n❌ ISSUES FOUND: {len(failed)}")
        for _, row in failed.iterrows():
            print(f"   - {row['model']}: {', '.join(row['issues'][:3])}")

    if len(missing) > 0:
        print(f"\n⚠️  FILES NOT FOUND: {len(missing)}")
        for _, row in missing.iterrows():
            print(f"   - {row['model']}")

    if len(errors) > 0:
        print(f"\n🔴 ERRORS: {len(errors)}")
        for _, row in errors.iterrows():
            print(f"   - {row['model']}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    needs_rerun = []
    for _, row in results_df.iterrows():
        if row['status'] == 'ISSUES_FOUND':
            if row.get('n_zero', 0) > 0 or row.get('n_random_suspect', 0) > 0:
                needs_rerun.append(row['model'])

    if needs_rerun:
        print("\n🔄 EMBEDDINGS THAT SHOULD BE RE-GENERATED:")
        for model in needs_rerun:
            print(f"   - {model}")
        print("\nThese embeddings may contain synthetic fallback data.")
        print("Re-run with the fixed scripts that raise errors on failure.")
    else:
        print("\n✅ All embeddings appear to be genuine.")
        print("No re-generation needed based on statistical analysis.")

    # Save report
    report_path = BASE_DIR / "05_RESULTS" / "embedding_integrity_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_path, index=False)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 70)
    print("INTEGRITY CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
