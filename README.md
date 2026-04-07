# HEAL: Hyperbolic Encoding with Geometry-Aligned Residual Latent Diffusion for Few-Shot Graph Classification

Anonymous code repository for KDD 2026 submission.

## Requirements

- Python >= 3.9
- PyTorch >= 1.12
- PyTorch Geometric >= 2.1
- PyYAML

```bash
pip install torch torch_geometric pyyaml
```

## Data

Datasets are bundled in `heal_datasets.zip`. Unzip to `data/` before running:

```bash
mkdir data && cd data && unzip ../heal_datasets.zip && cd ..
```

## Usage

Entry point is `main.py`.

```bash
# Full pipeline (train encoder → train diffusion → evaluate)
python main.py --dataset R52

# Encoder only, no diffusion augmentation
python main.py --dataset R52 --diffusion off

# Load pretrained encoder checkpoint
python main.py --dataset R52 --encoder <checkpoint_path>

# Load both pretrained encoder and diffusion
python main.py --dataset R52 --encoder <encoder_ckpt> --diffusion <diffusion_ckpt>
```

### Datasets
- `R52` (Graph-R52, 5-way)
- `COIL` (COIL-DEL, 20-way)
- `Letter_high` (Letter-High, 4-way)
- `Reddit` (Reddit, 4-way)

### Config Overrides
```bash
python main.py --dataset R52 --cfg model_curvature=4.0 diffusion_n_aug=4
```

Common options:
- `--cfg model_manifold_type=euclidean` — Euclidean (GIN) encoder
- `--cfg diffusion_generator_type=gaussian` — Gaussian noise instead of DDPM
- `--cfg model_curvature=<float>` — hyperbolic curvature
- `--cfg diffusion_n_aug=<int>` — synthetic samples per anchor
