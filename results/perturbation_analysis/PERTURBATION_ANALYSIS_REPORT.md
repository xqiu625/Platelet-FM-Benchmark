# Perturbation-to-Disease Mapping Analysis

## Identifying Therapeutic Targets for COVID-19 Severity Using Genome-Wide Perturb-seq

**Project:** Platelet-FM-Benchmark
**Author:** Xinru Qiu
**Last Updated:** December 15, 2025

---

## Summary

We mapped genome-wide perturbation data to COVID-19 disease severity states using foundation model embeddings to identify potential therapeutic targets. By projecting perturbation effects onto the recovery direction in embedding space, we identified 50 candidate genes whose knockdown shifts cells toward a recovered transcriptional state.

### Key Results

| Metric | Value |
|--------|-------|
| **Total Perturbations Analyzed** | 16,248 |
| **HEK293T Cells** | 88,434 |
| **HCT116 Cells** | 89,738 |
| **Therapeutic Candidates Identified** | 50 |
| **Foundation Model** | UCE (Universal Cell Embeddings) |

---

## Data Sources

### Perturbation Data

**X-Atlas/Orion Genome-Wide Perturb-seq Dataset**
> Huang et al. (2025). Genome-wide Perturb-seq Datasets via a Scalable Fix-Cryopreserve Platform for Training Dose-Dependent Biological Foundation Models. *bioRxiv*. [doi:10.1101/2025.06.11.659105](https://doi.org/10.1101/2025.06.11.659105)

- **Cell Lines:** HEK293T (88,434 cells), HCT116 (89,738 cells)
- **Genes Targeted:** 18,903 human genes
- **Platform:** FiCS (Fix-Cryopreserve-ScRNAseq)

### Disease Data

**COVID-19 Platelet Single-Cell Dataset**
> Qiu, X (2024). Deciphering Abnormal Platelet Subpopulations in COVID-19, Sepsis and Systemic Lupus Erythematosus. *IJMS*, 25(11), 5941.

- **Total Cells:** 47,808
- **Severity Classes:** Healthy, Mild, Moderate, Severe, Fatal, Recovered

---

## Methodology

### 1. Embedding Generation

Both datasets were embedded using UCE (Universal Cell Embeddings):
- Pre-trained on 36 million cells
- 1,280-dimensional embeddings
- Zero-shot evaluation (no fine-tuning)

### 2. Disease State Centroids

We computed centroid embeddings for each COVID-19 severity level:

| State | Cells | Description |
|-------|------:|-------------|
| Healthy | 3,205 | Control samples |
| Mild | 7,359 | Mild symptoms |
| Moderate | 4,330 | Moderate symptoms |
| Severe | 19,805 | Severe COVID |
| Fatal | 9,414 | Fatal cases |
| Recovered | 3,695 | Recovered patients |

### 3. Recovery Score Calculation

```
Recovery Direction = Centroid(recovered) - Centroid(severe)
Recovery Score = Perturbation_Effect · Recovery_Direction
```

Higher recovery scores indicate perturbations that shift cellular states toward recovery.

### 4. Therapeutic Candidate Selection

Candidates were selected based on:
1. High recovery score (top 1%)
2. Closest to recovered/healthy state
3. Excluding non-targeting controls

---

## Results

### Top 20 Therapeutic Candidates

| Rank | Gene | Recovery Score | Closest State | Cells | Function |
|:----:|------|:--------------:|:-------------:|:-----:|----------|
| 1 | **ICMT** | 27.15 | recovered | 5 | Protein prenylation |
| 2 | **ZNF766** | 27.10 | recovered | 6 | Transcription factor |
| 3 | **MED31** | 27.08 | recovered | 5 | Mediator complex |
| 4 | **ZFP30** | 27.02 | recovered | 5 | Zinc finger protein |
| 5 | **KRTAP21-3** | 27.00 | recovered | 5 | Keratin-associated |
| 6 | **DUSP11** | 26.98 | recovered | 6 | Dual-specificity phosphatase |
| 7 | **SLC28A1** | 26.96 | recovered | 5 | Nucleoside transporter |
| 8 | **ESD** | 26.95 | recovered | 5 | Esterase D |
| 9 | **AGPAT3** | 26.93 | recovered | 5 | Lipid biosynthesis |
| 10 | **B4GALT1** | 26.89 | recovered | 5 | Glycosyltransferase |
| 11 | **RAMP3** | 26.85 | recovered | 5 | Receptor activity modifier |
| 12 | **LMCD1** | 26.85 | recovered | 5 | LIM/cysteine-rich domain |
| 13 | **FAM83F** | 26.82 | recovered | 5 | Oncogene |
| 14 | **HEPACAM2** | 26.81 | recovered | 9 | Cell adhesion |
| 15 | **PXYLP1** | 26.81 | recovered | 5 | Phosphatase |
| 16 | **LILRB2** | 26.80 | recovered | 5 | Immune receptor |
| 17 | **H4C16** | 26.80 | recovered | 7 | Histone H4 |
| 18 | **ATP2B3** | 26.79 | recovered | 6 | Calcium ATPase |
| 19 | **VWCE** | 26.79 | recovered | 5 | Von Willebrand factor |
| 20 | **LYPD3** | 26.77 | recovered | 5 | GPI-anchored protein |

### Pathway Enrichment

| Pathway | Genes | Biological Relevance |
|---------|-------|---------------------|
| **Protein Modification** | ICMT, B4GALT1 | Post-translational regulation |
| **Transcription** | ZNF766, ZFP30, MED31 | Gene expression control |
| **Lipid Metabolism** | AGPAT3 | Membrane composition |
| **Immune Modulation** | LILRB2 | Immune checkpoint |
| **Coagulation** | VWCE | Platelet function |

### COVID-19 Relevant Candidates

1. **ICMT** (Isoprenylcysteine Carboxyl Methyltransferase)
   - Top-ranked candidate
   - Involved in Ras signaling and autophagy
   - Potential target for inflammation modulation

2. **LILRB2** (Leukocyte Immunoglobulin-Like Receptor B2)
   - Immune checkpoint receptor
   - Regulates macrophage and dendritic cell activation
   - Relevant to COVID-19 hyperinflammation

3. **VWCE** (Von Willebrand Factor C and EGF Domains)
   - Related to coagulation cascade
   - Directly relevant to COVID-19 coagulopathy
   - Platelet-specific significance

---

## Visualizations

### Perturbation Landscape

![Perturbation Landscape](../../figures/uce_perturbation_landscape.png)

UMAP visualization showing perturbation cells mapped to COVID-19 severity states. Colors indicate closest severity state for each perturbation.

### Therapeutic Rankings

![Therapeutic Rankings](../../figures/uce_therapeutic_rankings.png)

Bar chart showing top therapeutic candidates ranked by recovery score.

---

## Limitations

1. **Cell Type Mismatch:** Perturbations performed in HEK293T/HCT116, mapped to platelet disease states
2. **Indirect Inference:** Recovery scores are computational predictions, not experimental validations
3. **Single Foundation Model:** Results may vary with different embedding models (UCE vs STATE vs scGPT)

---

## Future Directions

1. **Multi-Model Comparison:** Compare UCE, STATE, and scGPT for perturbation mapping
2. **Experimental Validation:** Validate top candidates in relevant cell types
3. **Disease Expansion:** Apply to sepsis, lupus, and other inflammatory conditions
4. **Dose-Response:** Leverage X-Atlas dose-dependent data for therapeutic window prediction

---

## Data Availability

### Analysis Files

```
results/perturbation_analysis/
├── uce_perturbation_scores_*.csv       # All 16,248 perturbation scores
├── uce_therapeutic_candidates_*.csv    # Top 50 candidates
├── uce_analysis_report_*.txt           # Summary statistics
└── PERTURBATION_ANALYSIS_REPORT.md     # This document
```

### Figures

```
figures/
├── uce_perturbation_landscape.png/pdf  # UMAP visualization
└── uce_therapeutic_rankings.png/pdf    # Candidate rankings
```

---

## References

1. Huang et al. (2025). Genome-wide Perturb-seq Datasets via a Scalable Fix-Cryopreserve Platform. *bioRxiv*. [doi:10.1101/2025.06.11.659105](https://doi.org/10.1101/2025.06.11.659105)

2. Rosen et al. (2023). Universal Cell Embeddings. *bioRxiv*. [doi:10.1101/2023.11.28.568918](https://doi.org/10.1101/2023.11.28.568918)

3. Qiu, X (2024). Deciphering Abnormal Platelet Subpopulations in COVID-19, Sepsis and Systemic Lupus Erythematosus. *IJMS*, 25(11), 5941. [doi:10.3390/ijms25115941](https://doi.org/10.3390/ijms25115941)

---

**Contact:** Xinru Qiu (xinru.reina.qiu@gmail.com)
**GitHub:** [@xqiu625](https://github.com/xqiu625)
