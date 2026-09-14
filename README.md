# NeuralSOAP

**NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields**

NeuralSOAP is an all-atom 3D molecular generative framework that represents molecules as continuous multi-channel atomic occupancy fields and combines a neural-field latent code with a molecule-level Smooth Overlap of Atomic Positions (SOAP) embedding. The joint latent representation is modeled with Neural Empirical Bayes (NEB) and sampled using walk–jump sampling (WJS), followed by neural-field decoding and atom extraction.

## Method overview

For each molecule, NeuralSOAP combines:

- a learnable neural-field latent code `z`;
- a SOAP-derived structural embedding `s`.

The resulting representation is:

```text
c = [z; s]
```

The generation pipeline is:

1. Construct a continuous multi-channel atomic occupancy field.
2. Compute a molecule-level SOAP representation.
3. Learn a per-molecule latent modulation code.
4. Concatenate the latent and SOAP components.
5. Fit the joint latent distribution with Neural Empirical Bayes.
6. Sample the latent distribution with walk–jump dynamics.
7. Decode the sampled representation with the neural field.
8. Extract atom coordinates from the decoded occupancy field.

## Main components

- Continuous atomic occupancy fields
- Multiplicative Filter Network (MFN) decoder
- FiLM conditioning
- Molecule-level SOAP embedding
- Neural Empirical Bayes latent modeling
- Walk–jump sampling
- Grid-based atom extraction with local-maximum detection, peak refinement, non-maximum suppression, and minimum-distance filtering

## Datasets

The experiments in the paper use:

- **QM9** for the main unconditional 3D molecular generation benchmark
- **GEOM-Drugs** for evaluation on larger and more flexible drug-like molecules
- **MoleculeNet** tasks for downstream representation evaluation:
  - BBBP
  - BACE
  - Tox21
  - SIDER
  - ClinTox

## QM9 generation

The main QM9 evaluation reports:

- molecule stability
- atom stability
- validity
- uniqueness
- valency Wasserstein-1 distance
- atom total variation
- bond total variation
- bond-length Wasserstein-1 distance
- bond-angle Wasserstein-1 distance

For NeuralSOAP, the reported main results include:

- molecule stability: **99.1%**
- atom stability: **99.95%**
- validity: **100.0%**
- uniqueness: **99.6%**
- valency W1: **0.010**
- bond-length W1: **0.005**
- bond-angle W1: **1.115**

The main QM9 results are reported over three independent sampling runs.

## SOAP ablation

The paper includes two direct controls to quantify the contribution of SOAP.

### z-only

The SOAP component is removed, so generation uses only the neural-field latent code.

### shuffled SOAP

The correspondence between molecules and their SOAP representations is randomly permuted while the remaining experimental configuration is kept unchanged.

Reported QM9 results:

| Model | Stable molecule | Validity | Valency W1 | Bond-length W1 | Bond-angle W1 |
|---|---:|---:|---:|---:|---:|
| z-only | 98.7 | 99.8 | 0.014 | 0.007 | 1.30 |
| shuffled SOAP | 98.5 | 99.7 | 0.015 | 0.008 | 1.38 |
| NeuralSOAP | 99.1 | 100.0 | 0.010 | 0.005 | 1.115 |

These results show that correctly matched SOAP information improves the geometry-sensitive metrics under the reported setting.

## GEOM-Drugs

NeuralSOAP is also evaluated on GEOM-Drugs.

Reported results:

| Method | Validity | Stable molecule | Bond-angle W1 |
|---|---:|---:|---:|
| FuncMol | 100.0 ± 0.0 | 69.7 ± 0.2 | 2.49 ± 0.06 |
| NeuralSOAP | 98.4 ± 0.3 | 75.2 ± 0.9 | 2.03 ± 0.08 |

The FuncMol values are taken from the published baseline report, while the NeuralSOAP values are obtained from the NeuralSOAP evaluation.

## MoleculeNet evaluation

The paper reports a controlled downstream comparison using:

- D-MPNN
- SchNet
- SOAP-only + MLP
- NeuralSOAP

The 3D-dependent models use the same RDKit conformer for each molecule under the shared evaluation protocol. D-MPNN uses the corresponding molecule split with its native 2D graph representation.

Reported ROC-AUC values:

| Method | BBBP | BACE | Tox21 | SIDER | ClinTox |
|---|---:|---:|---:|---:|---:|
| D-MPNN | 0.88 | 0.82 | 0.78 | 0.65 | 0.86 |
| SchNet | 0.82 | 0.75 | 0.74 | 0.59 | 0.70 |
| SOAP-only + MLP | 0.84 | 0.79 | 0.75 | 0.56 | 0.78 |
| NeuralSOAP | 0.89 | 0.84 | 0.79 | 0.62 | 0.88 |

## Training details

The main QM9 configuration uses:

- MFN decoder layers: `6`
- MFN hidden width: `2048`
- conditioning code dimension: `1024`
- NEB denoiser residual blocks: `6`
- NEB denoiser hidden units: `4096`
- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-2`
- batch size: `1024`
- EMA decay: `0.9999`
- NEB noise level: `sigma = 0.9`
- WJS steps: `K = 2000`
- occupancy threshold: `0.4`
- atom-extraction grid resolution: `0.25 Å`
- QM9 training seed: `1234`

The reported SOAP configuration uses:

- cutoff radius: approximately `4.0 Å`
- `nmax = 8`
- `lmax = 6`

## Downstream adaptation

For the MoleculeNet experiments, the pretrained neural-field components are frozen and LoRA updates are applied to the linear layers that produce FiLM scale and shift parameters.

Reported settings:

- LoRA rank: `r = 8`
- LoRA scaling parameter: `alpha = 16`
- optimizer: `AdamW`
- learning rate: `1e-4`
- maximum epochs: `100`
- early-stopping patience: `10`

## Property distributions

For visualization of the QM9 property-distribution analysis, QED is shown in its native range, while SA and logP are linearly normalized to `[0, 1]`.

## Scope

The current implementation focuses on unconditional 3D molecular generation. Property-conditioned generation and differentiable structure extraction are natural extensions of the framework.

## Code availability

The implementation, configuration files, and evaluation scripts associated with the paper are provided in this repository.

Repository:

```text
https://github.com/codehxj/NeuralSOAP
```

## Citation

If you use NeuralSOAP, please cite:

```bibtex
@inproceedings{neuralsoap2026,
  title={NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields},
  booktitle={Asia Pacific Bioinformatics Conference},
  year={2026}
}
```
