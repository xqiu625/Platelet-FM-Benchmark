# 2025-06-FiCS-Perturb-seq-Platform-bioRxiv.md

## 📊 Paper Metadata
- **Title:** X-Atlas/Orion: Genome-wide Perturb-seq Datasets via a Scalable Fix-Cryopreserve Platform for Training Dose-Dependent Biological Foundation Models
- **Authors:** Ann C Huang, Tsung-Han S Hsieh, Jiang Zhu, Jackson Michuda, Ashton Teng, Soohong Kim, Elizabeth M Rumsey, Sharon K Lam, Ikenna Anigbogu, Philip Wright, Mohamed Ameen, Kwontae You, Christopher J Graves, Hyunsung John Kim, Adam J Litterman, Rene V Sit, Alex Blocker, Ci Chu
- **Publication:** bioRxiv preprint (2025)
- **Institution:** Xaira Therapeutics, Foresite Labs
- **Paper Link:** https://doi.org/10.1101/2025.06.11.659105
- **Code/Data:** [X-Atlas/Orion dataset available on FigShare](https://plus.figshare.com/articles/dataset/Processed_data_for_X-Atlas_Orion_Genome-wide_Perturb-seq_Datasets_via_a_Scalable_Fix-Cryopreserve_Platform_for_Training_Dose-Dependent_Biological_Foundation_Models/29190726)

## 🎨 Key Figures

### Figure 1: Industrialized Perturb-seq Platform Workflow
![Platform Overview](../../../paper-figures/fics-platform-workflow.png)

**Why this figure is exceptional:**
- **Comprehensive workflow visualization:** Shows complete end-to-end process from CRISPRi cell line generation to computational analysis with clear step-by-step breakdown
- **Technical innovation demonstration:** Highlights key FiCS platform innovations including fixation, FACS, cryopreservation, superloading, and automation
- **Scalability emphasis:** Illustrates how each component contributes to genome-wide screening capability and industrial-scale throughput
- **Implementation clarity:** Provides concrete technical details that enable reproducibility and adoption

**Design principles to mimic:**
- Hierarchical workflow layout with clear process flow
- Integration of biological and technical methodology steps
- Clear annotation of key innovations and improvements
- Quantitative specifications for each processing step

## 🔄 Key Scientific Insights

```python
### 1. Conceptual Innovation
- **FiCS Platform Development:** Revolutionary Fix-Cryopreserve-ScRNAseq platform addressing scalability and batch effect challenges in Perturb-seq
- **Dose-Dependent Modeling:** First demonstration of sgRNA abundance as quantitative proxy for perturbation strength enabling continuous variable analysis
- **Industrial Scale Achievement:** Largest publicly available Perturb-seq atlas with 8 million cells targeting all human protein-coding genes

### 2. Methodological Framework
- **FiCS Workflow:** DSP fixation + FACS enrichment + cryopreservation + superloading + automation for scalable perturbation screening
- **Dual-sgRNA System:** tRNA-based lentiviral delivery targeting same gene with reduced recombination rates (12.6% vs 26.1% without tRNA)
- **Platform Variants:**
  1. **DSP Fixation:** Reversible crosslinker preserving transcriptome correspondence with fresh cells
  2. **Fixation-Compatible FACS:** Zombie dye viability staining for live cell enrichment post-fixation
  3. **Superloading:** 100k cells/channel (5x standard) reducing cost and increasing throughput
  4. **Hamilton Automation:** Automated library preparation removing operator variability

### 3. Validation Strategy
**Comprehensive Evaluation Across:**
- **8 million cells** spanning HCT116 and HEK293T cell lines
- **18,903 genes** with dual-sgRNA targeting approach
- **Multiple quality metrics** including UMI depth, batch correlation, and biological pathway recovery
- **Comparison studies** against Replogle K562 and Tahoe-100M datasets
```

## 🔬 Critical Technical Details
```python
### 1. FiCS Platform Core Components

# Key workflow innovations:
- DSP Fixation: Lomant's Reagent (dithiobis(succinimidyl propionate)) for reversible crosslinking
- Viability Staining: Zombie NIR dye compatible with fixed cells for live/dead discrimination
- Cryopreservation: Bambanker medium + RNase inhibitor enabling 140+ day storage
- Superloading: 100k cells/channel vs 20k standard loading for 5x throughput increase

### 2. Data Processing and Quality Control
- **Cell Recovery:** 8 million cells with expected dual-sgRNA configuration from 30 million sequenced
- **Sequencing Depth:** Median 19,416 UMIs/cell (HCT116), 16,557 UMIs/cell (HEK293T)
- **Gene Detection:** 5,387 genes/cell (HCT116), 5,871 genes/cell (HEK293T)
- **Knockdown Efficiency:** 75.4% median KD (HCT116), 51.5% median KD (HEK293T)

### 3. Performance Metrics
- **Batch Consistency:** Median Spearman correlation 0.993 (HCT116), 0.988 (HEK293T) vs 0.967 (Replogle K562)
- **Effect Detection:** 48.36% (HCT116), 40.37% (HEK293T) perturbations with discernible transcriptomic effects
- **Dose Response:** Strong correlation (R=0.910-0.901) between sgRNA abundance and KD efficiency
- **Storage Stability:** Maintained RNA quality for 140 days in cryopreservation
```

## Baseline Models, Evaluation Metrics, and Datasets
```python
### Baseline Models (3 major comparisons)
- **Replogle K562:** Current gold standard genome-wide Perturb-seq dataset
- **Tahoe-100M:** Large-scale chemical perturbation atlas for comparison
- **Fresh Cell Controls:** Validation against non-fixed cell processing

### Evaluation Metrics
- **Primary:** UMIs per cell, genes per cell, batch correlation, knockdown efficiency
- **Transcriptomic Quality:** Binary classification accuracy, energy distance, pathway recovery
- **Dose Response:** sgRNA UMI correlation with KD efficiency, stratification capability

### Datasets
- **X-Atlas/Orion:** 8 million cells (HCT116: 3.4M, HEK293T: 4.5M) targeting 18,903 genes
- **Quality Control:** 3.25% (HCT116), 5.03% (HEK293T) cells used for pilot QC before full screening
- **Reference Standards:** Comparison against Replogle K562, STRING database, known protein complexes
```

## 💭 Critical Research Implications
```python
### 1. Methodological Impact
- **Scalability Revolution:** FiCS platform enables industrial-scale perturbation screening with consistent quality
- **Batch Effect Reduction:** Significant improvement in data consistency through fixation and automation
- **Cost Efficiency:** Superloading and automation reduce per-cell costs while maintaining quality

### 2. Foundation Model Relevance
- **Dose-Dependent Modeling:** sgRNA abundance enables continuous perturbation variables for more sophisticated AI models
- **Large-Scale Training Data:** 8 million high-quality cells provide substantial training dataset for biological foundation models
- **Causal Inference:** Systematic perturbation data enables training of models capable of predicting intervention outcomes

### 3. Biological Discovery Insights
- **Pathway Recovery:** Successful identification of known protein complexes validates biological relevance
- **Cell Line Differences:** CRISPRi efficiency varies significantly between cell types (75.4% vs 51.5% KD)
- **Component Dependencies:** sgRNA and TRIM28 expression levels predict CRISPRi performance across systems
```

## 💻 Computational Requirements
```python
### Hardware Specifications
- **Sequencing:** Illumina NovaSeq X Plus with 25B chemistry
- **Storage:** Large-scale data storage for 30+ million sequenced cells
- **Automation:** Hamilton Vantage liquid handler with multiple specialized modules
- **Flow Cytometry:** High-throughput sorters (SH800 Sony, CytoFlex BD) for FACS enrichment

### Software Environment
- **Alignment:** Cell Ranger 8.0.1 with GRCh38 2024-A reference
- **Analysis:** Python-based pipeline with scanpy, pandas, sklearn
- **sgRNA Calling:** Custom Gaussian mixture model for threshold determination
- **Quality Control:** Automated metrics calculation and batch correlation analysis

### Processing Times
- **Library Preparation:** Automated workflow comparable to manual (r=0.996 correlation)
- **Sequencing Strategy:** 50k reads/cell (gene expression), 5k reads/cell (CRISPR)
- **Cryopreservation:** Enables decoupling of cell harvest from library preparation
- **Scalability:** Platform handles 4-8 million cells per batch with consistent quality
``` 

## 🚀 Future Directions & Limitations
```python
### Potential Extensions
- **Additional Cell Types:** FiCS platform applicable to other cell lines and primary cells
- **Alternative Fixation:** FLEX and other fixation methods could provide similar benefits
- **Multi-modal Integration:** Combination with protein, chromatin accessibility, or other readouts

### Current Limitations
- **Cell Type Specificity:** CRISPRi efficiency varies significantly between cell lines
- **Multiplexing Constraints:** Poisson delivery and superloading reduce single-perturbation yield
- **sgRNA Sensitivity:** Detection algorithms may miss low-expression guides

### Open Questions
- How do fixation methods affect different cell types and perturbation responses?
- Can sgRNA abundance proxy extend to other perturbation modalities (CRISPRa, base editing)?
- What is the optimal balance between throughput and single-perturbation recovery?
```

## 📋 Implementation Checklist
```python
### For Reproducing Results
- [ ] DSP fixation protocol and reagent sourcing
- [ ] Hamilton Vantage automation setup with specified modules
- [ ] Flow cytometry capabilities for dual-positive sorting
- [ ] NovaSeq X sequencing capacity with 25B chemistry

### For Adapting to New Domains
- [ ] Cell line engineering for stable dCas9-KRAB expression
- [ ] sgRNA library design for target gene sets
- [ ] Cryopreservation optimization for specific cell types
- [ ] Computational pipeline adaptation for analysis requirements
```

## 🔗 Related Work & Context
- **Replogle et al. 2022:** Foundational genome-wide Perturb-seq establishing field standards
- **Dixit/Adamson 2016:** Original Perturb-seq methodology development
- **10x Genomics FLEX:** Alternative fixation-based single-cell approaches
- **Foundation Model Papers:** Context for AI-driven virtual cell development

## 🧩 Relevance to My Work

```python
### Key Overlaps with Project
- **Shared Task Type:** Large-scale perturbation profiling for foundation model training
- **Shared Methodology:** Single-cell transcriptomics with systematic genetic perturbations
- **Shared Data Type:** scRNA-seq with perturbation annotations and dose-response relationships

### What You Can Reuse or Build Upon
- **Platform architecture blueprint:** FiCS workflow for scalable perturbation screening
- **Dose-response framework:** sgRNA abundance as continuous perturbation variable
- **Quality control pipeline:** Comprehensive metrics for batch effects and biological validation
- **Automation strategies:** Hamilton Vantage protocols for reducing operator variability

### What This Paper Lacks (That You Can Contribute)
- **Multi-modal Integration:** No protein, chromatin, or metabolomic readouts alongside transcriptomics
- **Temporal Dynamics:** Static endpoint rather than time-course perturbation responses
- **Cell State Diversity:** Limited to two cell lines rather than diverse cell types or primary cells
- **Interaction Effects:** Focus on single-gene perturbations rather than combinatorial effects

### Potential for Citation or Collaboration
- **Cite for:** Scalable perturbation platform methodology, dose-response proxy validation, largest public Perturb-seq dataset
- **Contact for:** Platform implementation details, automation protocols, dataset access and integration
```

---
*Note based on analysis of: Huang et al. "X-Atlas/Orion: Genome-wide Perturb-seq Datasets via a Scalable Fix-Cryopreserve Platform for Training Dose-Dependent Biological Foundation Models" bioRxiv 2025*
