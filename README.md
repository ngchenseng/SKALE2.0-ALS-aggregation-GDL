# SKALE 2.0: Phase-Resolved Geometric Deep Learning for Disease-Associated Protein Aggregation

This repository provides the core Python implementation of **SKALE 2.0**, a phase-resolved geometric deep learning framework for modelling mutation-induced protein aggregation across distinct kinetic stages. SKALE 2.0 represents proteins as multimodal structural graphs and uses a Siamese equivariant graph neural network with phase-conditioned FiLM gating to compare wild-type and mutant protein states in a shared latent space.

The framework is designed to resolve how structural perturbations redistribute aggregation-relevant information across **nucleation** and **elongation**, with applications to disease-associated aggregation systems including **SOD1, TDP-43, MAPT, and PRNP**.

---

## Key Features

- **Geometric protein representation**  
  Builds residue-level structural graphs from Cα coordinates, local spatial connectivity, solvent accessibility, hydrogen-bond features, conformational dynamics, and ESM2 residue embeddings.

- **Siamese WT-to-mutant encoding**  
  Processes wild-type and mutant structures with shared EGNN weights to quantify mutation-induced latent displacement.

- **Phase-resolved aggregation modelling**  
  Uses learnable phase tokens and FiLM conditioning to separate nucleation-linked and elongation-linked structural determinants.

- **Multimodal feature integration**  
  Integrates AlphaFold-derived or structure-derived descriptors, hydrogen-bond organization, NMA/fluctuation features, and frozen ESM2 embeddings.

- **Kinetic supervision and auxiliary reconstruction**  
  Supports regression of aggregation kinetic parameters together with auxiliary feature reconstruction to stabilize representation learning.

- **Reproducible training workflow**  
  Includes manifest construction, feature loading, graph construction, model training, validation split, checkpointing, and QC plots.

---

## Repository Contents

```text
skale2_core_training.py
README.md
```

The current script contains the **core machine-learning framework and training pipeline** corresponding to Cells 1–15 of the original SKALE 2.0 workflow. Downstream figure-generation, saturation mutagenesis, saliency visualization, and inverse-design analyses are not included in this core training script.

---

## Method Overview

SKALE 2.0 encodes each protein structure as a graph:

- nodes represent Cα residues;
- edges connect residues within a spatial radius;
- node features include structural, physicochemical, dynamic, and sequence-embedding descriptors.

Wild-type and mutant structures are passed through a shared Siamese EGNN backbone. Phase-specific representations are then generated using FiLM-modulated nucleation and elongation tokens. The resulting latent embeddings are used to estimate mutation-induced structural displacement and train phase-conditioned aggregation-related outputs.

---

## Installation

The script automatically installs the main Python dependencies when executed in Google Colab:

```bash
pip install fair-esm biopython scikit-learn
```

Core packages used by the workflow include:

```text
torch
numpy
pandas
matplotlib
biopython
scikit-learn
fair-esm
```

---

## Data Requirements

The script expects local or Google Drive folders containing:

- PDB files for wild-type and mutant protein structures;
- per-residue SASA feature tables;
- hydrogen-bond feature tables;
- NMA or fluctuation feature tables;
- optional kinetic data tables.

By default, the script uses the following Google Drive paths:

```text
/content/drive/MyDrive/Structural_analysis/
/content/drive/MyDrive/Structural_analysis3/
/content/drive/MyDrive/Structural_analysis4/
/content/drive/MyDrive/Structural_analysis6/
```

These paths can be overridden using environment variables:

```bash
export SKALE2_OUTDIR="/path/to/output"
export SKALE2_DIR_SOD1="/path/to/SOD1"
export SKALE2_DIR_TDP43="/path/to/TDP43"
export SKALE2_DIR_MAPT="/path/to/MAPT"
export SKALE2_DIR_PRNP="/path/to/PRNP"
```

---

## Running the Core Training Script

In Google Colab or a local Python environment, run:

```bash
python skale2_core_training.py
```

The script will:

1. install required dependencies;
2. mount Google Drive when running in Colab;
3. construct a manifest of available protein structures and feature files;
4. load residue-level structural and sequence features;
5. build radius-based protein graphs;
6. initialize the SKALE 2.0 Siamese EGNN model;
7. train the model using available WT-to-mutant pairs and kinetic labels;
8. save checkpoints and QC plots.

Outputs are written by default to:

```text
/content/drive/MyDrive/SKALE2.0/
```

---

## Model Components

The core implementation includes:

- `EGNNLayer` for equivariant message passing;
- `PhaseFiLM` for nucleation and elongation phase conditioning;
- `SKALE2` model class with:
  - global aggregation-risk head,
  - kinetic-parameter regression head,
  - auxiliary reconstruction head;
- manifest and sample builders;
- ESM2 embedding cache;
- masked loss function;
- pair-level train/validation split;
- checkpointing and QC visualization.

---

## Scope and Notes

This repository currently provides the **core training implementation** of SKALE 2.0. The broader manuscript includes additional downstream analyses, including latent manifold visualization, phase-specific saliency, in silico saturation mutagenesis, motif grammar analysis, and suppressor-design workflows. These analyses are conceptually linked to the core framework but are not included in this minimal training release.

The present workflow is focused on disease-associated aggregation systems. Experimental validation in the corresponding manuscript is centered on SOD1, while computational evaluation spans SOD1, TDP-43, MAPT, and PRNP.

---

## Suggested Repository Description

A phase-resolved geometric deep learning framework for modelling disease-associated protein aggregation using Siamese EGNNs, FiLM phase tokens, structural graphs, ESM2 embeddings, and kinetic supervision.

---

## Citation

If you use this code, please cite the associated manuscript once available:

```text
Sio et al. A phase-resolved geometric deep learning framework maps structural determinants of disease-associated protein aggregation and guides suppressor design.
```

---

## Contact

For questions about the SKALE 2.0 framework, please contact:

**Chen Seng Ng**  
School of Science, Monash University Malaysia  
Email: ng.chenseng@monash.edu
