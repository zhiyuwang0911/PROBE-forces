# PROBE: Post-hoc Reliability frOm Backbone Embeddings

Official code for:

> **Knowing when to trust machine-learned interatomic potentials**  
> Shams Mehdi, Ilkwon Cho, Olexandr Isayev  

---

## Overview

PROBE attaches a lightweight binary classifier to the **frozen** per-atom
representations of a pretrained MLIP, learning to answer one question:
*is this prediction reliable?*

It requires no modification to the underlying model, adds <1% inference
overhead, and generalizes across architectures — demonstrated here on
**AIMNet2** and **MACE-OFF23**.

---

## Repository Structure

```
PROBE/
├── probe/
│   ├── model.py            # PROBEModel architecture
│   ├── train.py            # training loop, loss, evaluation
│   ├── metrics.py          # accuracy, MCC, F1, calibration
│   └── backends/
│       ├── aimnet2.py      # AIMNet2 data loading & batch processing
│       └── mace.py         # MACE data loading & batch processing
├── train_aimnet2.py        # runnable training script for AIMNet2
├── train_mace.py           # runnable training script for MACE (energy only)
├── train_mace_multitask.py # runnable training script for MACE (energy + force)
├── infer_mace_multitask.py # inference for multitask checkpoint + test.xyz
├── environment_aimnet2.yml
└── environment_mace.yml
```

---

## Installation

**For MACE:**
```bash
conda env create -f environment_mace.yml
conda activate probe_mace
```

**For MACE + cuEquivariance (CUDA acceleration, PyTorch ≥ 2.4):**
```bash
module load cuda/12.1.1   # on HPC clusters, before conda create
conda env create -f environment_mace_cueq.yml
conda activate probe_mace_cueq
python train_mace_multitask.py --enable-cueq
```

**For AIMNet2:**
```bash
conda env create -f environment_aimnet2.yml
conda activate probe_aimnet2

# AIMNet2 must be installed from source:
git clone https://github.com/isayevlab/AIMNet2
pip install -e AIMNet2/
```

---

## Training PROBE

### On MACE-OFF23

1. Edit the `CONFIG` block in `train_mace.py`:

```python
CONFIG = {
    'mace_model_path': '/path/to/MACE-OFF23_large.model',
    'train_xyz':       '/path/to/train.xyz',
    'test_xyz':        '/path/to/test.xyz',
    'output_dir':      './probe_mace_outputs',
    ...
}
```

2. Run:

```bash
python train_mace.py
```

### Multitask on MACE-OFF23 (energy + force)

`train_mace_multitask.py` extends PROBE to classify **energy reliability**,
**per-atom force reliability**, and **structure-level force reliability**
(structure force predictions are mean-aggregated from per-atom logits; no extra
head). MACE energy and forces are computed via a live forward pass.

1. Edit the `CONFIG` block in `train_mace_multitask.py` (paths, batch size,
   architecture, etc.).

2. Run:

```bash
python train_mace_multitask.py
```

**CUDA acceleration (optional):**

```bash
python train_mace_multitask.py --enable-cueq
```

**Loss weights (optional):** the total objective is a weighted sum of three
cross-entropy terms — energy (`L_E`), per-atom force (`L_Fa`), and
structure-level force (`L_Fs`). Set weights via CLI (overrides `CONFIG`) or in
`CONFIG`:

```bash
python train_mace_multitask.py \
  --lambda-energy 1.0 \
  --lambda-force-atom 1.0 \
  --lambda-force-mol 0.3
```

| Flag | `CONFIG` key | Default | Task |
|------|----------------|---------|------|
| `--lambda-energy` | `lambda_energy` | `1.0` | Structure energy reliability |
| `--lambda-force-atom` | `lambda_force_atom` | `1.0` | Per-atom force reliability |
| `--lambda-force-mol` | `lambda_force_mol` | `1.0` | Structure force reliability (derived from atom logits) |

`L_Fs` is computed from the same per-atom force head as `L_Fa`, so lowering
`lambda_force_mol` (e.g. `0.3`) is often reasonable when the structure-level
force task is easier or redundant. Train and validation loss use the same
weighted sum for direct comparison.

The best checkpoint is saved to `output_dir/best_multitask_model_<timestamp>.pt`.
Every epoch writes two resumable files:
- `output_dir/checkpoint_epoch_XXXX.pt` — per-epoch snapshot
- `output_dir/last_checkpoint.pt` — always the latest (overwrite)

Also `best_multitask_model_<timestamp>.pt` when validation improves.

Resume without re-scanning error boundaries:

```bash
python train_mace_multitask.py --resume
python train_mace_multitask.py --resume ./outputs/checkpoint_epoch_0012.pt
```

### On AIMNet2

1. Edit the `CONFIG` block in `train_aimnet2.py`:

```python
CONFIG = {
    'checkpoint':    '/path/to/aimnet2_checkpoint.pt',
    'arch_yaml':     '/path/to/aimnet2.yaml',
    'inference_cfg': '/path/to/UQ_aimnet2_config.yaml',
    'output_dir':    './probe_aimnet2_outputs',
    ...
}
```

2. Run:

```bash
python train_aimnet2.py
```

Both scripts auto-detect the class boundary from the training-set error
distribution (50th percentile by default) and save the best checkpoint to
`output_dir/best_model_<timestamp>.pt`.

---

## Architecture

```
Frozen MLIP backbone
        │
        ▼  {h_i} ∈ R^d  per-atom embeddings
  Atom Encoder MLP
  (d → 256, LayerNorm, GELU, dropout=0.1)
        │
        ▼  (+ partial charge injection for AIMNet2)
  Multi-Head Self-Attention  (32 heads × 8 dims)
        │
        ▼
  Masked mean-pool ∥ masked max-pool ∥ energy ∥ N_atoms  ∈ R^514
        │
        ▼  linear projection
  Molecular embedding  ∈ R^256
        │
        ▼
  Classifier MLP  [256 → 128 → 32 → 2]
        │
        ▼
  P(reliable),  P(unreliable)
```

Total trainable parameters: ~567K

---

## Extending to a New MLIP

To apply PROBE to a different MLIP:

1. Write a `process_batch_fn(batch, device)` that returns:
   `(atom_feats [B,N,D], atom_mask [B,N], pred_energy [B], true_energy [B], n_atoms [B])`

2. Instantiate `PROBEModel(backbone_dim=D)`.

3. Call `run_training(model, process_batch_fn, ...)`.

No other changes are needed.

---

## Inference (multitask energy + force)

```bash
python infer_mace_multitask.py \
  --mace-model /path/to/MACE-OFF23_large.model \
  --checkpoint /path/to/best_multitask_model_YYYYMMDD_HHMMSS.pt \
  --test-xyz /path/to/test.xyz \
  --output-dir ./probe_multitask_inference
```

Writes `predictions_structure.csv`, `predictions_atom.csv`, `predictions.npz`, and `metrics.json` (if reference labels are present).

## Inference and Atom Importance


```python
import torch
from probe.model import PROBEModel

model = PROBEModel(backbone_dim=256)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# atom_feats: [B, N, 256], atom_mask: [B, N] bool
with torch.no_grad():
    logits = model(atom_feats, atom_mask, energy=pred_energy)
    probs  = torch.softmax(logits, dim=-1)     # P(reliable), P(unreliable)
    importance = model.get_atom_importance(atom_feats, atom_mask)  # [B, N]
```

---

## License

MIT — see [LICENSE](LICENSE).

This repository extends the original [PROBE](https://github.com/isayevlab/PROBE) code
(Isayev Lab, Carnegie Mellon University). Multitask energy/force modifications are
Copyright (c) 2026 Zhiyu Wang.
