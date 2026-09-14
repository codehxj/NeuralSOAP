# NeuralSOAP

**NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields**

NeuralSOAP is an all-atom 3D molecular generative framework that combines continuous atomic occupancy fields with a molecule-level Smooth Overlap of Atomic Positions (SOAP) representation. A shared neural-field decoder models molecular geometry, while Neural Empirical Bayes (NEB) and walk–jump sampling (WJS) are used to model and sample the joint latent representation.

## Overview

NeuralSOAP represents a molecule as a continuous multi-channel atomic occupancy field. Each molecule is associated with:

- a learnable neural-field latent code `z`;
- a SOAP-derived structural embedding `s`.

The two components are combined as

`c = [z; s]`

and modeled jointly in latent space. Generation proceeds by sampling the joint latent distribution with NEB/WJS, decoding the sampled representation through the neural field, and extracting atom coordinates from the resulting occupancy field.

## Main components

- Continuous multi-channel atomic occupancy fields
- Multiplicative Filter Network (MFN) decoder
- FiLM conditioning
- Molecule-level SOAP structural representation
- Neural Empirical Bayes latent modeling
- Walk–jump sampling
- Grid-based atom extraction with peak refinement and non-maximum suppression

## Datasets and experiments

The current study includes:

### QM9

Unconditional 3D molecular generation on QM9, including evaluation of:

- molecular stability
- atom stability
- validity
- uniqueness
- valency Wasserstein-1 distance
- atom and bond total variation
- bond-length Wasserstein-1 distance
- bond-angle Wasserstein-1 distance

### SOAP ablation

The contribution of SOAP is evaluated using two direct controls:

- **z-only**: generation without the SOAP component;
- **shuffled SOAP**: the correspondence between molecules and SOAP representations is randomly permuted while the remaining experimental configuration is kept unchanged.

These controls are evaluated using the same sampling, atom-extraction, and evaluation settings as the full NeuralSOAP model.

### GEOM-Drugs

NeuralSOAP is additionally evaluated on GEOM-Drugs to assess generation on larger and more flexible drug-like molecules.

Reported metrics include:

- validity
- molecular stability
- bond-angle Wasserstein-1 distance

### MoleculeNet

A controlled downstream evaluation is conducted on:

- BBBP
- BACE
- Tox21
- SIDER
- ClinTox

The comparison includes:

- D-MPNN
- SchNet
- SOAP-only + MLP
- NeuralSOAP

For the 3D-dependent models, the same RDKit conformer is used for each molecule under the shared evaluation protocol.

## Training

NeuralSOAP is trained in two stages.

### Stage 1: neural-field auto-decoder

The shared neural-field decoder, SOAP projection, and per-molecule latent codes are optimized using occupancy-field reconstruction.

### Stage 2: latent generative modeling

The concatenated latent representations are modeled using Neural Empirical Bayes. Molecular samples are generated using walk–jump sampling and then decoded into continuous occupancy fields.

The main QM9 configuration uses:

- MFN decoder: 6 layers
- hidden width: 2048
- conditioning dimension: 1024
- NEB denoiser: 6 residual blocks
- denoiser hidden dimension: 4096
- optimizer: AdamW
- learning rate: `1e-3`
- weight decay: `1e-2`
- batch size: `1024`
- EMA decay: `0.9999`
- NEB noise level: `sigma = 0.9`
- WJS steps: `K = 2000`
- occupancy threshold: `0.4`
- atom-extraction grid resolution: `0.25 Å`
- random seed: `1234`

The reported SOAP configuration uses:

- cutoff radius: approximately `4.0 Å`
- `nmax = 8`
- `lmax = 6`

## MoleculeNet adaptation

For downstream adaptation, pretrained neural-field components are frozen and LoRA updates are applied to the linear layers producing the FiLM scale and shift parameters.

The reported configuration uses:

- LoRA rank: `r = 8`
- LoRA scaling: `alpha = 16`
- optimizer: AdamW
- learning rate: `1e-4`
- maximum epochs: `100`
- early-stopping patience: `10`

Additional implementation details, configuration files, and evaluation scripts are provided in this repository.


## Reproducibility

To reproduce the reported experiments, use the configuration files and scripts corresponding to:

- QM9 main generation
- QM9 z-only ablation
- QM9 shuffled-SOAP ablation
- GEOM-Drugs generation
- MoleculeNet controlled evaluation

All reported comparisons should use the dataset partitions, conformer inputs, and evaluation settings specified by the corresponding experiment configuration.

## Scope

The current implementation focuses on unconditional 3D molecular generation. Property-conditioned generation, differentiable atom extraction, and further scaling to broader chemical spaces remain directions for future work.

## Citation

If you use NeuralSOAP, please cite the APBC 2026 paper:

```bibtex
@inproceedings{neuralsoap2026,
  title={NeuralSOAP: Structure-Aware 3D Molecular Generation with SOAP-Guided Neural Fields},
  booktitle={Asia Pacific Bioinformatics Conference},
  year={2026}
}
```

## License

Please add the license used by this repository here.
