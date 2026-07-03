"""
Train multitask PROBE on MACE-OFF23 (energy + force reliability).

Structure-level force labels use mean per-atom force component MAE.
Structure-level force *predictions* mean-aggregate per-atom force logits
(no extra structure force head).

Edit the CONFIG block below, then run:
    python train_mace_multitask.py
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.model import MultitaskPROBEModel
from probe.backends.mace import (
    load_mace, get_z_table,
    train_val_split_loader,
    process_batch_mace,
    process_batch_mace_multitask,
    scan_force_error_boundaries,
)
from probe.train import run_multitask_training, compute_error_boundary

# ============================================================
# Configuration — edit these paths before running
# ============================================================
CONFIG = {
    # Paths
    'mace_model_path':   '/path/to/MACE-OFF23_large.model',
    'train_xyz':         '/path/to/train.xyz',
    'test_xyz':          '/path/to/test.xyz',
    'output_dir':        './probe_mace_multitask_outputs',

    # Device
    'device':            'cuda' if torch.cuda.is_available() else 'cpu',
    'ev_to_kcalmol':     23.06,

    # Data
    'batch_size':        256,
    'valid_fraction':    0.1,

    # Error boundaries (50th percentile = balanced classes on train)
    'error_boundary_percentile': 50,

    # Multitask loss weights
    'lambda_energy':       1.0,
    'lambda_force_atom':   1.0,
    'lambda_force_mol':    1.0,

    # Training
    'lr':                5e-5,
    'weight_decay':      1e-4,
    'epochs':            1000,
    'early_stopping_patience': 10,
    'scheduler_patience':      5,
    'scheduler_factor':        0.9,
    'min_lr':            5e-6,
    'gradient_clip_norm':1.0,

    # Architecture (backbone_dim auto-detected from MACE)
    'atom_encoder_hidden':       [256, 128],
    'atom_encoder_output_dim':   256,
    'mol_attention_heads':       32,
    'classifier_hidden':         [256, 128, 32],
    'atom_force_head_hidden':    [128, 32],
    'dropout':                   0.1,

    # Evaluation
    'high_conf_cutoffs': {0: 0.8, 1: 0.8},
}


def main():
    device = CONFIG['device']

    # 1. Load frozen MACE backbone
    extractor = load_mace(CONFIG['mace_model_path'], device)
    z_table   = get_z_table(extractor)
    r_max     = float(extractor.mace_model.r_max)

    # 2. Load data (90/10 train/val split from train_xyz)
    print("Loading data...")
    train_loader, val_loader = train_val_split_loader(
        CONFIG['train_xyz'], z_table, r_max,
        CONFIG['batch_size'], CONFIG['valid_fraction'],
    )

    # 3. Energy boundary from training set
    print("Computing energy error distribution on training set...")
    errors_kcal = []
    for batch in tqdm(train_loader, desc='Scanning energy errors'):
        _, _, pred_e, true_e, _ = process_batch_mace(batch, device, extractor)
        err = torch.abs(true_e - pred_e)
        valid = ~torch.isnan(err)
        errors_kcal.extend((err[valid].cpu().numpy() * CONFIG['ev_to_kcalmol']).tolist())

    boundary_kcal = compute_error_boundary(
        np.array(errors_kcal), CONFIG['error_boundary_percentile'])
    boundary_ev = boundary_kcal / CONFIG['ev_to_kcalmol']
    error_bins_e = torch.tensor([0.0, boundary_ev], device=device)

    # 4. Force boundaries from training set (per-atom + mean-atom structure)
    print("Computing force error distribution on training set...")
    boundary_f_atom, boundary_f_mol = scan_force_error_boundaries(
        train_loader, device, extractor, CONFIG['error_boundary_percentile'])
    error_bins_f_atom = torch.tensor([0.0, boundary_f_atom], device=device)
    error_bins_f_mol  = torch.tensor([0.0, boundary_f_mol], device=device)

    # 5. Build multitask PROBE model
    model = MultitaskPROBEModel(
        backbone_dim=extractor.feat_dim,
        atom_encoder_hidden=CONFIG['atom_encoder_hidden'],
        atom_encoder_output_dim=CONFIG['atom_encoder_output_dim'],
        mol_attention_heads=CONFIG['mol_attention_heads'],
        classifier_hidden=CONFIG['classifier_hidden'],
        atom_force_head_hidden=CONFIG['atom_force_head_hidden'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Multitask PROBE parameters: {total_params:,}")

    # 6. Train
    process_fn = lambda batch, dev: process_batch_mace_multitask(batch, dev, extractor)
    history = run_multitask_training(
        model=model,
        process_batch_fn=process_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        error_bins_e=error_bins_e,
        error_bins_f_atom=error_bins_f_atom,
        error_bins_f_mol=error_bins_f_mol,
        device=device,
        output_dir=CONFIG['output_dir'],
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        epochs=CONFIG['epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        scheduler_patience=CONFIG['scheduler_patience'],
        scheduler_factor=CONFIG['scheduler_factor'],
        min_lr=CONFIG['min_lr'],
        gradient_clip_norm=CONFIG['gradient_clip_norm'],
        lambda_energy=CONFIG['lambda_energy'],
        lambda_force_atom=CONFIG['lambda_force_atom'],
        lambda_force_mol=CONFIG['lambda_force_mol'],
        high_conf_cutoffs=CONFIG['high_conf_cutoffs'],
    )

    print(f"\nTraining complete. Best epoch: {history['best_epoch']}")
    print(f"Checkpoint saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
