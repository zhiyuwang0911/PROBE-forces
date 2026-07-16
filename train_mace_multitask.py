"""
Train multitask PROBE on MACE-OFF23 (energy + force reliability).

Structure-level force labels use mean per-atom force component MAE.
Structure-level force *predictions* mean-aggregate per-atom force logits
(no extra structure force head).

MACE energy/forces are computed via live forward pass (not read from extxyz).

Edit the CONFIG block below, then run:
    python train_mace_multitask.py
    python train_mace_multitask.py --enable-cueq   # NVIDIA CUDA acceleration
    python train_mace_multitask.py --lambda-energy 1.0 --lambda-force-atom 1.0 --lambda-force-mol 0.3
    python train_mace_multitask.py --resume        # continue from output_dir/last_checkpoint.pt
    python train_mace_multitask.py --cache-mace    # cache MACE after first pass (default on)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.model import MultitaskPROBEModel
from probe.backends.mace import (
    load_mace, get_z_table,
    train_val_split_loader,
    CachedMACEProcessor,
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
    'enable_cueq':       False,   # True: NVIDIA cuEquivariance (CUDA + extra pkgs)
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
    # last_checkpoint.pt is always rewritten each epoch (for HPG TIMEOUT resume).
    # checkpoint_every > 0 also keeps checkpoint_epoch_XXXX.pt every N epochs.
    'checkpoint_every':  0,

    # MACE cache: first pass fills RAM; later epochs reuse (no MACE).
    # Set mace_cache_dir (or --mace-cache-dir) to also persist on disk for resume.
    'cache_mace':        True,
    'mace_cache_dir':    None,  # None = RAM only; path = also write {idx}.pt files

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train multitask PROBE on MACE-OFF23')
    parser.add_argument(
        '--enable-cueq', action='store_true',
        help='Enable NVIDIA cuEquivariance CUDA acceleration for MACE '
             '(overrides CONFIG enable_cueq)',
    )
    parser.add_argument(
        '--lambda-energy', type=float, default=None,
        help='Loss weight for energy reliability (default: CONFIG lambda_energy)',
    )
    parser.add_argument(
        '--lambda-force-atom', type=float, default=None,
        help='Loss weight for per-atom force reliability '
             '(default: CONFIG lambda_force_atom)',
    )
    parser.add_argument(
        '--lambda-force-mol', type=float, default=None,
        help='Loss weight for structure-level force reliability '
             '(default: CONFIG lambda_force_mol)',
    )
    parser.add_argument(
        '--resume', nargs='?', const='AUTO', default=None,
        help='Resume training. With no path, uses '
             'CONFIG[output_dir]/last_checkpoint.pt',
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=None,
        help='Also keep checkpoint_epoch_XXXX.pt every N epochs '
             '(0 = only rewrite last_checkpoint.pt each epoch; '
             'default: CONFIG checkpoint_every)',
    )
    parser.add_argument(
        '--cache-mace', action='store_true', default=None,
        help='Cache MACE embeddings/preds after first pass (default: CONFIG)',
    )
    parser.add_argument(
        '--no-cache-mace', action='store_true',
        help='Disable MACE caching (recompute every epoch)',
    )
    parser.add_argument(
        '--mace-cache-dir', type=str, default=None,
        help='Also persist MACE cache to this directory for resume '
             '(default: CONFIG mace_cache_dir, else RAM only)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override CONFIG output_dir (checkpoints written here)',
    )
    return parser.parse_args()


def _resolve_lambda(cli_value, config_key: str) -> float:
    """CLI value overrides CONFIG when provided."""
    if cli_value is not None:
        return cli_value
    return CONFIG[config_key]


def main():
    args = parse_args()
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    device = CONFIG['device']
    enable_cueq = CONFIG['enable_cueq'] or args.enable_cueq
    lambda_energy = _resolve_lambda(args.lambda_energy, 'lambda_energy')
    lambda_force_atom = _resolve_lambda(args.lambda_force_atom, 'lambda_force_atom')
    lambda_force_mol = _resolve_lambda(args.lambda_force_mol, 'lambda_force_mol')
    checkpoint_every = (args.checkpoint_every
                        if args.checkpoint_every is not None
                        else CONFIG.get('checkpoint_every', 1))

    resume_path = None
    if args.resume is not None:
        resume_path = (Path(CONFIG['output_dir']) / 'last_checkpoint.pt'
                       if args.resume == 'AUTO' else Path(args.resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    print(
        f"Loss weights: lambda_energy={lambda_energy}, "
        f"lambda_force_atom={lambda_force_atom}, "
        f"lambda_force_mol={lambda_force_mol}"
    )

    # 1. Load frozen MACE backbone
    extractor = load_mace(
        CONFIG['mace_model_path'], device, enable_cueq=enable_cueq)
    z_table   = get_z_table(extractor)
    r_max     = float(extractor.mace_model.r_max)

    # 2. Load data (90/10 train/val split from train_xyz)
    print("Loading data...")
    train_loader, val_loader = train_val_split_loader(
        CONFIG['train_xyz'], z_table, r_max,
        CONFIG['batch_size'], CONFIG['valid_fraction'],
    )

    # 3. MACE processor (optional cache: fill on first pass, reuse later)
    use_cache = CONFIG.get('cache_mace', True)
    if args.no_cache_mace:
        use_cache = False
    elif args.cache_mace:
        use_cache = True

    if use_cache:
        cache_dir = args.mace_cache_dir or CONFIG.get('mace_cache_dir')
        if cache_dir:
            print(f"MACE cache enabled (RAM + disk) → {cache_dir}")
        else:
            print("MACE cache enabled (RAM only; set --mace-cache-dir to persist)")
        process_fn = CachedMACEProcessor(
            extractor, compute_force=True, cache_dir=cache_dir)
    else:
        from probe.backends.mace import process_batch_mace_multitask
        print("MACE cache disabled (recompute every epoch)")
        process_fn = lambda batch, dev: process_batch_mace_multitask(
            batch, dev, extractor)

    # 4. Error boundaries (skip full MACE scans when resuming)
    if resume_path is not None:
        print(f"Loading error bins from resume checkpoint {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        error_bins_e = torch.tensor(
            resume_ckpt['error_bins_energy'], device=device, dtype=torch.float32)
        error_bins_f_atom = torch.tensor(
            resume_ckpt['error_bins_force_atom'], device=device, dtype=torch.float32)
        error_bins_f_mol = torch.tensor(
            resume_ckpt['error_bins_force_mol'], device=device, dtype=torch.float32)
        print(f"  energy bins={error_bins_e.tolist()}")
        print(f"  force_atom bins={error_bins_f_atom.tolist()}")
        print(f"  force_mol bins={error_bins_f_mol.tolist()}")
    else:
        # Single train pass with forces fills energy+force bins and warms cache.
        print("Computing energy + force error distributions on training set...")
        errors_kcal = []
        for batch in tqdm(train_loader, desc='Scanning energy errors'):
            (_, _, pred_e, true_e, _, _, _) = process_fn(batch, device)
            err = torch.abs(true_e - pred_e)
            valid = ~torch.isnan(err)
            errors_kcal.extend(
                (err[valid].cpu().numpy() * CONFIG['ev_to_kcalmol']).tolist())

        boundary_kcal = compute_error_boundary(
            np.array(errors_kcal), CONFIG['error_boundary_percentile'])
        boundary_ev = boundary_kcal / CONFIG['ev_to_kcalmol']
        error_bins_e = torch.tensor([0.0, boundary_ev], device=device)

        print("Computing force error distribution on training set...")
        boundary_f_atom, boundary_f_mol = scan_force_error_boundaries(
            train_loader, device, extractor, CONFIG['error_boundary_percentile'],
            process_batch_fn=process_fn)
        error_bins_f_atom = torch.tensor([0.0, boundary_f_atom], device=device)
        error_bins_f_mol  = torch.tensor([0.0, boundary_f_mol], device=device)
        if use_cache and isinstance(process_fn, CachedMACEProcessor):
            print(f"  MACE cache after boundary scan: {len(process_fn)} structures "
                  f"(hits={process_fn.hits}, misses={process_fn.misses})")

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
        lambda_energy=lambda_energy,
        lambda_force_atom=lambda_force_atom,
        lambda_force_mol=lambda_force_mol,
        high_conf_cutoffs=CONFIG['high_conf_cutoffs'],
        resume_path=str(resume_path) if resume_path else None,
        checkpoint_every=checkpoint_every,
    )

    print(f"\nTraining complete. Best epoch: {history['best_epoch']}")
    print(f"Checkpoint saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
