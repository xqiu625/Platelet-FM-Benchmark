# Foundation Models for Single-Cell Transcriptomics: A Comprehensive Comparison

## Overview

This document compares five foundation models for single-cell RNA sequencing analysis, including their architectures, training approaches, and performance on our COVID-19 severity prediction benchmark.

---

## Model Summary Table

| Feature | Geneformer | UCE | scGPT | TranscriptFormer | STATE |
|---------|------------|-----|-------|------------------|-------|
| **Publication** | Nature 2023 | bioRxiv 2023 | Nature Methods 2024 | bioRxiv 2025 | Nature 2025 |
| **Institution** | Broad/Harvard | Stanford/CZ BioHub | U Toronto | CZ Initiative | Arc Institute |
| **Training Cells** | 30M | 36M | 33M | 112M | 167M (SE) + 100M (ST) |
| **Embedding Dims** | 1,152 | 1,280 | 512 | 2,048 | Variable |
| **Parameters** | ~10M | 650M | ~50M | 444-542M | 600M (SE) |
| **Architecture** | BERT-style Encoder | Transformer + ESM2 | GPT-style Decoder | Autoregressive | Set-based Transformer |
| **Species** | Human only | 8+ species | Human | 12 species | Human |
| **Primary Focus** | Network biology | Cross-species | Multi-omics | Generative atlas | Perturbation |

---

## Detailed Model Descriptions

### 1. Geneformer (Nature 2023)

**Core Innovation:** Transfer learning for network biology using rank-based gene encoding.

**Architecture:**
- 6-layer transformer encoder (BERT-style)
- 256 embedding dimensions internally, 1,152 output
- 4 attention heads per layer
- Input: 2,048 genes max (covers 93% of cells)
- Masking rate: 15% (self-supervised)

**Key Features:**
- **Rank Value Encoding:** Genes ranked by expression normalized to median across corpus
  - Deprioritizes housekeeping genes
  - Elevates cell-state-defining genes
  - Robust to batch effects
- **Transfer Learning:** Fine-tune with as few as 884 cells
- **Network Learning:** Attention weights encode gene network hierarchy

**Training Data:**
- Genecorpus-30M: 29.9 million human cells
- 561 datasets, droplet-based only
- Excludes malignant cells

**Best For:**
- Rare disease analysis with limited samples
- Gene dosage sensitivity prediction
- Therapeutic target discovery
- Network dynamics understanding

**Limitations:**
- Human-only (no cross-species)
- 2,048 gene limit truncates some cells
- Rank encoding loses precise expression values

---

### 2. UCE - Universal Cell Embeddings (bioRxiv 2023)

**Core Innovation:** Cross-species cell representation using protein language model embeddings.

**Architecture:**
- 33-layer transformer encoder
- 650 million parameters
- 1,280-dimensional cell embeddings
- Uses ESM2 protein embeddings for genes

**Key Features:**
- **"Bags of RNA" Approach:** Weighted gene sampling with protein embeddings
- **Zero-Shot Capability:** No fine-tuning required for cell type classification
- **Cross-Species:** Works across 8+ species without homolog mapping
- **Batch Correction:** Handles batch effects and experimental artifacts

**Training Data:**
- Integrated Mega-scale Atlas (IMA): 36 million cells
- CellXGene Census: 33.9 million cells
- Multiple species validation datasets

**Best For:**
- Cross-species cell type annotation
- Zero-shot classification
- Dataset integration
- Novel cell type discovery

**Limitations:**
- Transcriptomics only (no multi-omics)
- Large model requires significant compute
- No generative capabilities

---

### 3. scGPT (Nature Methods 2024)

**Core Innovation:** Generative pre-training with simultaneous cell and gene representation learning.

**Architecture:**
- Stacked transformer blocks
- 512-dimensional embeddings
- GPT-style with specialized attention masks
- `<cls>` token for cell-level representation
- FlashAttention for efficiency

**Key Features:**
- **Multi-Task Learning:** Cell annotation, perturbation prediction, batch integration, multi-omics
- **Gene Expression Prediction (GEP):** Masked gene prediction objective
- **Elastic Cell Similarity (ECS):** Fine-tuning objective for similar cells
- **Attention-Based Networks:** Gene regulatory network inference from attention weights

**Training Data:**
- CELLxGENE: 33 million human cells
- Diverse tissue types and conditions

**Best For:**
- Perturbation response prediction
- Multi-omics integration (RNA + ATAC + protein)
- Batch correction
- Gene regulatory network inference

**Limitations:**
- Smaller embedding dimension (512) may limit capacity
- Requires fine-tuning for best performance
- Human-only pre-training

---

### 4. TranscriptFormer (bioRxiv 2025)

**Core Innovation:** Cross-species generative model spanning 1.53 billion years of evolution.

**Architecture:**
- 12 transformer layers, 16 attention heads
- 2,048 model dimension
- 444-542 million parameters
- Dual decoder: Categorical gene selection + Zero-truncated Poisson counts
- ESM-2 embeddings for cross-species gene alignment

**Key Features:**
- **Expression-Aware Attention:** Count-based bias in attention computation
- **Cross-Species Generalization:** Works across 12 species (coral to human)
- **Generative Capability:** Can generate realistic transcriptomes
- **Zero-Shot Species Transfer:** >0.65 F1 even at 685M years divergence

**Training Data:**
- 112 million cells from CZ CELLxGENE + curated atlases
- 12 species spanning metazoa
- 3.5 trillion training tokens

**Best For:**
- Cross-species cell atlas construction
- Evolutionary conservation studies
- Zero-shot generalization to new species
- Generative cell modeling

**Limitations:**
- Not specialized for perturbation prediction
- Limited handling of technical batch effects
- Requires massive compute (1000 H100 GPUs for training)

---

### 5. STATE (Nature 2025)

**Core Innovation:** Set-based perturbation modeling with population-level learning.

**Architecture:**
- **State Embedding (SE):** 600M parameter encoder-decoder
- **State Transition (ST):** LLaMA/GPT2-based transformer
- Set-based attention across cell populations (32-512 cells)
- Maximum Mean Discrepancy (MMD) loss

**Key Features:**
- **Set-Based Attention:** First to use self-attention across cell sets for perturbations
- **Multi-Scale Modeling:** Combines population-level (ST) with cell-level (SE)
- **Zero-Shot Transfer:** Cross-dataset generalization
- **Multi-Modal Perturbations:** Chemical, genetic, and signaling

**Training Data:**
- SE: 167 million cells (Arc scBaseCount, CZ CELLxGENE, Tahoe-100M)
- ST: 100M+ perturbed cells across 70+ contexts

**Best For:**
- Perturbation response prediction
- Drug effect modeling
- Cross-context generalization
- Virtual screening

**Limitations:**
- Requires large compute resources
- Performance depends on available perturbation data
- Limited validation on primary tissues

---

## Our Benchmark Results: COVID-19 Severity Prediction

*Last Updated: January 4, 2026*

### Task Definition
- **Binary Classification:** Severe (severe + fatal) vs Non-severe (healthy + mild + moderate + recovered)
- **3-Class Classification:** Healthy vs Mild/Moderate vs Severe/Fatal
- **Dataset:** ~47,800 platelet single-cell transcriptomes
- **Evaluation:** 5-fold stratified cross-validation

### Fine-Tuned Performance

| Rank | Model | Embedding Dims | Binary AUC | 3-Class AUC | Status |
|------|-------|----------------|:----------:|:-----------:|--------|
| 🥇 | **STATE** | 2,058 | **0.944** | **0.948** | ✅ Verified |
| 🥈 | UCE | 1,280 | 0.910 | 0.912 | ✅ Verified |
| 🥉 | Geneformer | 1,152 | 0.845 | 0.851 | ✅ Verified |
| 4 | TranscriptFormer | 2,048 | 0.838 | *TBD* | 🔄 Re-running |
| 5 | scGPT | 512 | 0.735 | 0.738 | ✅ Verified |

### Zero-Shot Performance

| Model | Binary AUC | 3-Class AUC |
|-------|:----------:|:-----------:|
| STATE | 0.895 | 0.909 |
| UCE | 0.877 | 0.896 |
| Geneformer | 0.813 | 0.826 |
| scGPT | 0.775 | 0.747 |

### Baseline Comparison

| Method | Binary AUC | Notes |
|--------|:----------:|-------|
| STATE (fine-tuned) | **0.944** | Best overall |
| UCE (fine-tuned) | 0.910 | |
| **Raw XGBoost** | 0.897 | Strong baseline |
| STATE (zero-shot) | 0.895 | |
| UCE (zero-shot) | 0.877 | |
| PCA_500 + LogReg | 0.850 | |

### Robustness Analysis

#### Sample Efficiency (% drop at 5% training data)
| Rank | Model | Drop | Interpretation |
|:----:|-------|:----:|----------------|
| 1 | **UCE** | **-7%** | Most robust to limited data |
| 2 | scGPT | -4% | |
| 3 | STATE | -13% | |
| 4 | Geneformer | -15% | |

#### Batch-Shift Generalization (Train COVID → Test Sepsis)
| Rank | Model | AUC |
|:----:|-------|:---:|
| 1 | **STATE** | **0.789** |
| 2 | UCE | 0.750 |
| 3 | Geneformer | 0.730 |
| 4 | scGPT | 0.524 |

#### Gene Dropout Robustness (% drop at 70% dropout)
| Model | 0% | 70% | Drop |
|-------|:--:|:---:|:----:|
| **UCE** | 0.876 | 0.861 | **-1.7%** |
| STATE | 0.893 | 0.862 | -3.4% |

### Clinical Utility Metrics (Binary Task)

| Model | AUC-ROC | AUC-PR | Sens@90%Spec | Cohen's κ |
|-------|:-------:|:------:|:------------:|:---------:|
| **STATE** | **0.894** | **0.934** | **71.7%** | **0.611** |
| UCE | 0.876 | 0.923 | 67.4% | 0.568 |
| TranscriptFormer | 0.838 | 0.895 | 58.5% | 0.501 |
| Geneformer | 0.824 | 0.888 | 56.4% | 0.475 |
| scGPT | 0.776 | 0.836 | 40.8% | 0.402 |

### Statistical Significance

All pairwise comparisons: **p < 0.0001**

| Comparison | Δ AUC | Effect Size |
|------------|:-----:|:-----------:|
| STATE vs UCE | +0.034 | Medium |
| STATE vs Geneformer | +0.099 | Large |
| UCE vs Geneformer | +0.065 | Large |

### Key Findings

1. **STATE achieves best overall performance** (0.944 AUC)
   - Set-based attention captures perturbation-relevant features
   - Best batch generalization (COVID→Sepsis: 0.789 AUC)
   - Highest clinical utility (71.7% sensitivity @ 90% specificity)

2. **UCE shows superior robustness**
   - Most sample-efficient (-7% drop at 5% data)
   - Most dropout-robust (-1.7% at 70% gene dropout)
   - Strong zero-shot performance (0.877 AUC)

3. **Strong XGBoost baseline** (0.897 AUC)
   - Foundation models provide 2-5% improvement over baselines
   - Validates that embeddings capture clinically relevant information

4. **scGPT underperforms on this task**
   - Smaller embedding dimension (512) may limit capacity
   - Near-random on batch-shift (0.524 AUC)
   - May be better suited for perturbation tasks

5. **Embedding dimension is not the only factor**
   - STATE (2,058 dims) > TranscriptFormer (2,048 dims)
   - UCE (1,280 dims) >> scGPT (512 dims)

---

## Architecture Comparison

### Input Representation

| Model | Gene Encoding | Expression Encoding | Special Tokens |
|-------|---------------|---------------------|----------------|
| Geneformer | Rank-based (position in sorted list) | Implicit in rank | None |
| UCE | ESM2 protein embeddings | Weighted sampling | None |
| scGPT | Learnable gene tokens | Binned values | `<cls>`, `<pad>` |
| TranscriptFormer | ESM2 + learnable | Zero-truncated Poisson | Start/end tokens |
| STATE | ESM2 + soft binning | Expression-aware | Set tokens |

### Attention Mechanisms

| Model | Attention Type | Special Features |
|-------|----------------|------------------|
| Geneformer | Full self-attention | Network hierarchy in weights |
| UCE | Standard transformer | Gene-cell interactions |
| scGPT | Masked (generative) | Specialized masks for generation |
| TranscriptFormer | Expression-aware | Count-biased attention |
| STATE | Set-based bidirectional | Cross-cell attention |

### Pre-training Objectives

| Model | Objective | Description |
|-------|-----------|-------------|
| Geneformer | Masked gene prediction | Predict 15% masked genes |
| UCE | Contrastive learning | Similar cells closer in embedding space |
| scGPT | Generative prediction | Predict gene expression from context |
| TranscriptFormer | Autoregressive generation | Next-gene prediction with counts |
| STATE | Distributional matching | MMD loss for perturbation effects |

---

## Use Case Recommendations

### For COVID-19/Disease Severity Prediction
**Recommended: STATE** (primary) or **UCE** (if robustness is critical)
- STATE: Best overall performance (AUC 0.944), best clinical utility (71.7% sens @ 90% spec)
- UCE: Most robust to limited samples and gene dropout; strong zero-shot (0.877 AUC)
- Both significantly outperform baselines

### For Perturbation/Drug Response
**Recommended: STATE**
- Specifically designed for perturbation prediction
- 54% improvement over baselines on perturbation discrimination
- Handles chemical, genetic, and signaling perturbations
- Best batch-shift generalization (0.789 AUC COVID→Sepsis)

### For Cross-Species Analysis
**Recommended: UCE or TranscriptFormer**
- UCE: Simpler, zero-shot, 8+ species
- TranscriptFormer: More species (12), generative capability

### For Rare Disease with Limited Samples
**Recommended: UCE** (primary) or **Geneformer** (for network biology)
- UCE: Only -7% drop at 5% training data (most sample-efficient)
- Geneformer: Transfer learning with as few as 884 cells; network biology focus

### For Multi-Omics Integration
**Recommended: scGPT**
- Handles RNA + ATAC + protein simultaneously
- Batch correction across modalities
- Gene regulatory network inference
- Note: Underperforms on single-modality disease prediction

### For Clinical Translation
**Recommended: STATE**
- Highest sensitivity at 90% specificity (71.7%)
- Best Cohen's kappa (0.611)
- Best AUC-PR (0.934)

---

## Computational Requirements Comparison

| Model | Training Hardware | Inference | Memory |
|-------|-------------------|-----------|--------|
| Geneformer | 12x V100 (3 days) | Single GPU | ~10GB |
| UCE | Multi-GPU cluster | Single GPU | ~20GB |
| scGPT | Multi-GPU | Single GPU | ~10GB |
| TranscriptFormer | 1000x H100 (weeks) | Single GPU | ~30GB |
| STATE | Multi-H100 (SE), A100 (ST) | GPU recommended | ~50GB |


---

## References

1. Theodoris et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618, 616-624.
2. Rosen et al. (2023). Universal Cell Embeddings: A Foundation Model for Cell Biology. *bioRxiv*.
3. Cui et al. (2024). scGPT: Toward Building a Foundation Model for Single-cell Multi-omics. *Nature Methods*.
4. Pearce et al. (2025). TranscriptFormer: A Cross-Species Generative Cell Atlas. *bioRxiv*.
5. Adduri et al. (2025). Predicting cellular responses to perturbation with State. *Nature*.

---

## Biological Insights from Model Interpretability

### Top Biomarkers for Severe COVID (LogReg Coefficients)

| Gene | Direction | Biological Function |
|------|-----------|---------------------|
| **CST7** | Severe | Cystatin F - immune regulation |
| PF4 | Severe | Platelet factor 4 - coagulation |
| FTH1 | Severe | Ferritin heavy chain - iron metabolism |
| LYZ | Protective | Lysozyme - antimicrobial |
| CCL5 | Protective | Chemokine - immune recruitment |

### Therapeutic Candidates (Perturbation Recovery Analysis)

**STATE-identified targets:**
1. SLC25A38 (recovery: 1.052)
2. TRAPPC2 (recovery: 1.039)
3. EVI2A (recovery: 0.987)

**UCE-identified targets:**
1. CELSR1 (recovery: 0.703)
2. FBXO10 (recovery: 0.701)
3. VWA2 (recovery: 0.670)

---

*Document created: December 3, 2025*
*Last updated: January 4, 2026*
*Project: 302006 - COVID-19 Disease Severity Prediction*
