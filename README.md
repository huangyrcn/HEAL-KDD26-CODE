# HEAL-KDD26-CODE

Implementation of HEAL: Hyperbolic Graph Data Augmentation for Few-Shot Learning.

## Usage

```bash
python train.py --dataset <dataset> --cfg <key=value>
```

Common configs:
- `--dataset`: Letter_high, COIL, R52, Reddit
- `--model_aug`: augmentation method (ddpm, none)
- `--n_way`, `--n_shot`: few-shot settings
- `--cfg model_hyp=true`: enable hyperbolic encoder
- `--cfg model_aug=true`: enable DDPM augmentation

Example:
```bash
python train.py --dataset R52 --n_way 5 --n_shot 10 --cfg model_aug=true
```