# NeuralSOAP

**Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields**

Official implementation of **NeuralSOAP**, an all-atom 3D molecular generative framework that combines continuous neural fields, SOAP-derived structural representations, Neural Empirical Bayes (NEB), and walk-jump sampling (WJS).

The paper **“NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields”** has been accepted for an oral presentation at **APBC 2026: the 24th Asia Pacific Bioinformatics Conference**.

---

## Overview

NeuralSOAP represents 3D molecules as continuous multi-channel atomic occupancy fields and generates new molecular structures in a compact latent space.

For each molecule, the model combines:

- a learnable neural-field latent code \(z\), which captures global molecular variation;
- a SOAP-derived structural embedding \(s\), which summarizes local atomic environments.

The joint latent representation is

\[
c = [z; s].
\]

The empirical distribution of these joint codes is modeled using **Neural Empirical Bayes**, and new latent samples are generated through **walk-jump sampling**. The sampled latent representation is then decoded by a coordinate-based neural field and converted into atomistic 3D structures.

The current study focuses on **unconditional 3D molecular generation on QM9**.

---

## Framework

```text
Input 3D molecule
       |
       v
Continuous multi-channel atomic occupancy field
       |
       +-------------------------------+
       |                               |
       v                               v
Neural-field latent code z        SOAP descriptor
                                       |
                                       v
                                SOAP embedding s
       |                               |
       +---------------+---------------+
                       |
                       v
                   c = [z; s]
                       |
                       v
             Neural Empirical Bayes
                       |
                       v
               Walk-jump sampling
                       |
                       v
              Neural-field decoder
                       |
                       v
          Multi-channel occupancy field
                       |
                       v
                 Atom extraction
                       |
                       v
              Generated 3D molecule
```

---

## Main Features

- **Continuous molecular representation**  
  Molecules are modeled as continuous multi-channel atomic occupancy fields rather than fixed-resolution voxel grids.

- **SOAP-guided structural representation**  
  Smooth Overlap of Atomic Positions (SOAP) descriptors provide rotation-invariant information about local atomic environments.

- **Latent-space generation**  
  Molecular generation is performed in a compact latent space rather than directly in the high-dimensional field space.

- **Neural Empirical Bayes**  
  A denoising model learns the Gaussian-smoothed empirical distribution of latent molecular representations.

- **Walk-jump sampling**  
  Langevin dynamics explores the smoothed latent distribution before a denoising jump maps the sample back toward the clean latent manifold.

- **Continuous neural-field decoding**  
  The sampled latent code is decoded into an atomic occupancy field from which atom coordinates and atom types can be extracted.

---

## Repository Structure

```text
NeuralSOAP/
├── configs/               # Hydra configuration files
├── dataset/               # Dataset processing and molecular field construction
├── models/                # Neural-field and latent generative models
├── utils/                 # Training, evaluation, and molecular utilities
├── precompute_soap.py     # SOAP feature preprocessing
├── train_nf.py            # Train the molecular neural field
├── eval_nf.py             # Evaluate neural-field reconstruction
├── infer_codes.py         # Infer latent molecular codes
├── train_fm.py            # Train the latent generative model
├── sample_fm.py           # Generate molecules using latent sampling
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/codehxj/NeuralSOAP.git
cd NeuralSOAP
```

We recommend using a dedicated Python environment:

```bash
conda create -n neuralsoap python=3.10
conda activate neuralsoap
```

Install PyTorch according to your CUDA environment.

Typical dependencies include:

```bash
pip install torch torchvision
pip install lightning
pip install hydra-core omegaconf
pip install torchmetrics
pip install numpy scipy scikit-learn
pip install ase dscribe rdkit
pip install tqdm
```

A CUDA-enabled GPU is recommended for training and large-scale molecular generation.

---

## Dataset

The experiments reported in the paper use the **QM9** molecular dataset.

QM9 contains approximately 133,000 small organic molecules with DFT-optimized 3D geometries and the following atom types:

```text
C, H, O, N, F
```

The molecular field therefore contains one occupancy channel for each atom type.

Before training, prepare the QM9 dataset according to the data-loading format expected by the code and place the processed files in the configured dataset directory.

Dataset paths and preprocessing options can be modified through the Hydra configuration files under:

```text
configs/
```

---

## Training and Generation

NeuralSOAP is trained in two stages.

### Stage 1: Train the Neural Field

Train the continuous molecular neural-field representation:

```bash
python train_nf.py
```

The neural field learns to reconstruct molecular occupancy fields from molecular latent codes and SOAP-derived structural information.

Configuration options can be modified through the corresponding Hydra configuration file.

Hydra arguments can also be overridden from the command line, for example:

```bash
python train_nf.py seed=1234
```

---

### SOAP Feature Precomputation

SOAP descriptors can be precomputed using:

```bash
python precompute_soap.py
```

SOAP represents local atomic neighborhoods through smooth atomic densities and rotation-invariant power-spectrum features.

The atom-level SOAP descriptors are aggregated into a molecule-level structural representation and projected to a compact embedding \(s\).

---

### Stage 2: Infer Latent Codes

After training the neural field, infer latent molecular representations:

```bash
python infer_codes.py
```

The resulting latent codes are used to construct the empirical latent distribution for generative modeling.

---

### Stage 3: Train the Latent Generative Model

Train the latent generative model:

```bash
python train_fm.py
```

The model learns the distribution of the joint molecular representation

\[
c=[z;s].
\]

The denoising objective follows the Neural Empirical Bayes formulation.

Given a clean latent code \(c\), Gaussian noise is added:

\[
y=c+\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,\sigma^2I).
\]

A denoising network is trained to recover the clean code:

\[
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\|c-\hat{c}_{\theta}(c+\epsilon)\|_2^2
\right].
\]

Under Gaussian corruption, the denoiser also provides an estimate of the score of the smoothed latent distribution.

---

### Stage 4: Generate Molecules

Generate new molecular samples using:

```bash
python sample_fm.py
```

Generation follows the walk-jump procedure:

1. initialize a noisy latent sample;
2. perform Langevin dynamics in the smoothed latent distribution;
3. apply a denoising jump;
4. obtain the joint code \(c=[z;s]\);
5. decode the code using the neural field;
6. extract atom coordinates and atom types from the decoded occupancy field.

---

## Evaluation

Neural-field reconstruction can be evaluated using:

```bash
python eval_nf.py
```

The paper evaluates generated molecules using commonly reported 3D molecular generation metrics, including:

- molecule stability;
- atom stability;
- chemical validity;
- uniqueness;
- valency Wasserstein distance;
- atom-type total variation;
- bond-type total variation;
- bond-length Wasserstein distance;
- bond-angle Wasserstein distance;
- sampling time.

The main QM9 evaluation uses 10,000 generated molecules.

---

## Main QM9 Results

Representative results reported in the paper are shown below.

| Method | Stable Atom ↑ | Stable Molecule ↑ | Valid ↑ | Unique ↑ |
|---|---:|---:|---:|---:|
| NeuralSOAP | 99.95 ± 0.02 | 99.1 ± 0.10 | 100.0 | 99.6 ± 0.10 |

NeuralSOAP achieves competitive performance on QM9 while using a representation and sampling strategy that differs from point-cloud diffusion and flow-based molecular generators.

The reported error bars summarize variation across three independent sampling runs and should be interpreted descriptively rather than as formal evidence of statistical significance.

---

## Representation Diagnostics

In addition to standard generation benchmarks, the paper analyzes the learned molecular representation through:

- latent-space interpolation;
- t-SNE visualization;
- linear probing of QM9 molecular properties;
- SOAP-based local-geometry similarity;
- strain-energy distributions;
- atom-count and ring-size distributions;
- QED, synthetic accessibility, and logP distributions;
- auxiliary downstream molecular property classification.

These experiments are intended primarily as **representation diagnostics**.

They should not be interpreted as controlled evidence isolating the independent causal contribution of SOAP.

---

## Generation Objective and Controllability

The current NeuralSOAP model performs **unconditional molecular generation**.

The neural-field reconstruction objective learns a continuous representation of molecular geometry, while the Neural Empirical Bayes objective models the empirical distribution of latent molecular codes.

The current implementation does **not** explicitly optimize or condition generation on target molecular properties such as:

- QED;
- logP;
- binding affinity;
- toxicity;
- biological activity;
- target-specific drug response.

Property-directed generation could be introduced in future work by learning a conditional latent distribution such as

\[
p(c \mid y),
\]

where \(y\) represents a desired molecular property or biological condition.

---

## Scope and Limitations

The current implementation corresponds to the experimental setting described in the paper.

Important limitations include:

1. **QM9-focused evaluation**  
   The current experiments are restricted to relatively small molecules from QM9.

2. **Unconditional generation**  
   The present model does not directly perform property-conditioned or target-conditioned molecular generation.

3. **SOAP contribution**  
   SOAP is incorporated as a geometry-aware component of the latent representation. Direct causal ablations, including a \(z\)-only model and shuffled-SOAP variants, are important for quantitatively isolating its independent contribution.

4. **Atom extraction**  
   The continuous neural field must eventually be converted into discrete atoms. Extraction therefore depends on field evaluation and post-processing parameters.

5. **Drug-like molecular scale**  
   The current QM9 experiments should not be interpreted as evidence that the framework has already been validated on large, flexible, drug-like molecules.

Future work will investigate:

- direct SOAP ablations;
- larger molecular datasets;
- drug-like molecules;
- conditional molecular generation;
- differentiable atom extraction;
- structure- and property-guided molecular optimization.

---

## Reproducibility

Training and experimental parameters are controlled through Hydra configuration files.

For reproducible experiments, we recommend recording:

- Git commit hash;
- Python version;
- PyTorch version;
- CUDA version;
- GPU model;
- random seed;
- complete Hydra configuration;
- dataset split;
- model checkpoint;
- sampling configuration.

Example:

```bash
python train_nf.py seed=1234
```

The same principle applies to latent-model training and sampling.

---

## Paper

**NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields**

Accepted for oral presentation at:

**APBC 2026 — 24th Asia Pacific Bioinformatics Conference**

Hsinchu, Taiwan, 2026.

If you use this code or find this work useful, please cite the paper.

```bibtex
@inproceedings{neuralsoap2026,
  title     = {NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields},
  booktitle = {Proceedings of the 24th Asia Pacific Bioinformatics Conference},
  year      = {2026}
}
```

The final bibliographic information will be updated after publication of the conference proceedings.

---

## Code Availability

The source code for NeuralSOAP is publicly available at:

https://github.com/codehxj/NeuralSOAP

The repository contains the main components for:

- neural-field training;
- SOAP feature computation;
- latent-code inference;
- latent generative modeling;
- walk-jump sampling;
- molecular reconstruction and evaluation.

---

## Citation

If this repository contributes to your research, please cite the NeuralSOAP paper.

---

## Contact

For questions about the implementation, reproducibility, or experiments, please open an issue in this repository.
