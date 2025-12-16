# Key Benchmark Findings

## Last Updated: December 15, 2025

---

## Critical Finding: Raw XGBoost Beats All Foundation Models!

| Rank | Method | Binary AUC | Type |
|:----:|--------|:----------:|:----:|
| **1** | **Raw_XGBoost** | **0.897** | Baseline |
| 2 | STATE | 0.894 | Foundation Model |
| 3 | Raw_LogReg | 0.878 | Baseline |
| 4 | UCE | 0.876 | Foundation Model |

**This is a provocative finding for the single-cell foundation model community!**

---

## Core Benchmark Results (All Tasks)

| Model | Binary AUC | 3-Class AUC | 6-Class AUC | Avg AUC | Rank |
|-------|:----------:|:-----------:|:-----------:|:-------:|:----:|
| **STATE** | **0.894** | **0.893** | **0.894** | **0.894** | 1 |
| UCE | 0.876 | 0.885 | 0.888 | 0.883 | 2 |
| TranscriptFormer | 0.838 | 0.851 | 0.849 | 0.846 | 3 |
| Geneformer | 0.824 | 0.833 | 0.821 | 0.826 | 4 |
| scGPT | 0.833 | 0.747 | 0.810 | 0.797 | 5 |

---

## Clinical Metrics (Binary Classification)

| Model | AUC-ROC | AUC-PR | Sens@90%Spec | Spec@90%Sens | Cohen's Kappa |
|-------|:-------:|:------:|:------------:|:------------:|:-------------:|
| **STATE** | **0.894** | **0.934** | **0.717** | **0.650** | **0.611** |
| UCE | 0.876 | 0.923 | 0.674 | 0.591 | 0.568 |
| TranscriptFormer | 0.838 | 0.895 | 0.585 | 0.508 | 0.501 |
| Geneformer | 0.824 | 0.888 | 0.564 | 0.458 | 0.475 |
| scGPT | 0.776 | 0.836 | 0.408 | 0.421 | 0.402 |

**Clinical Insights:**
- STATE achieves 71.7% sensitivity at 90% specificity - clinically meaningful
- scGPT struggles clinically (only 40.8% sensitivity at 90% specificity)

---

## Subsampling Robustness (Binary)

| Model | 100% | 50% | 20% | 10% | 5% | Drop |
|-------|:----:|:---:|:---:|:---:|:--:|:----:|
| **STATE** | **0.894** | 0.880 | 0.842 | 0.793 | 0.781 | -0.113 |
| UCE | 0.877 | 0.870 | 0.855 | 0.835 | 0.814 | **-0.063** |
| TranscriptFormer | 0.841 | 0.830 | 0.802 | 0.773 | 0.749 | -0.092 |
| scGPT | 0.775 | 0.764 | 0.754 | 0.747 | 0.741 | **-0.034** |
| Geneformer | 0.823 | 0.812 | 0.778 | 0.735 | 0.695 | -0.128 |

**Key Findings:**
- UCE is most robust among top performers (-0.063 drop at 5%)
- scGPT most robust overall (-0.034) but starts lower
- STATE degrades more (-0.113) - needs more training data

---

## Key Insights for Paper

### 1. Raw Expression + XGBoost is King
- Raw_XGBoost (0.897) beats all foundation models
- Task-specific optimization may outperform general embeddings
- Provocative for the single-cell foundation model community

### 2. Foundation Model Value is Limited
- Only STATE (0.894) is competitive with XGBoost
- UCE barely beats Raw_LogReg (0.876 vs 0.878)
- TranscriptFormer, scGPT, Geneformer all underperform PCA baselines

### 3. STATE Still Wins Among Foundation Models
- Best foundation model across all tasks
- Best clinical metrics (AUC-PR=0.934, Sens@90%Spec=0.717)
- Remarkably consistent (0.893-0.894 across tasks)

### 4. Robustness Trade-offs
| Model | Performance | Data Efficiency | Trade-off |
|-------|:-----------:|:---------------:|-----------|
| STATE | Best (0.894) | Moderate (-0.113) | High perf, needs more data |
| UCE | Good (0.877) | Good (-0.063) | Balanced |
| scGPT | Low (0.775) | Best (-0.034) | Low perf, very robust |

---

## Paper Narrative (Draft)

> **Key Message:** While foundation models for single-cell biology show promise, their value for specific downstream tasks remains limited. In COVID-19 severity prediction from platelet transcriptomics, a simple gradient boosting model on raw gene expression (AUC=0.897) outperforms all tested foundation models including STATE (0.894), UCE (0.876), and others. Only STATE provides marginally competitive performance, while three foundation models (TranscriptFormer, scGPT, Geneformer) underperform simple PCA baselines. This suggests that task-specific feature engineering may still be more valuable than general-purpose embeddings for clinical prediction tasks.

---

## Extended Analysis Status (In Progress)

| Analysis | Binary | 3-Class | 6-Class | Status |
|----------|:------:|:-------:|:-------:|:------:|
| Core Benchmark | ✅ | ✅ | ✅ | **COMPLETE** |
| PCA Baseline | ✅ | 🔄 | 🔄 | Running |
| Clinical Metrics | ✅ | 🔄 | - | Running |
| Subsampling | ✅ | 🔄 | - | Running |
| Gene Dropout | 🔄 | 🔄 | - | Running |
| Embedding Ablations | 🔄 | 🔄 | - | Running |

---

_Last updated: December 15, 2025_
