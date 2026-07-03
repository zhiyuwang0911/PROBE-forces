#!/usr/bin/env python3
"""
Build labeling summary CSV from extxyz + live MACE predictions.

Reads structure inputs and reference energy/forces from extxyz, runs MACE
forward passes for predicted energy/forces, computes errors, assigns binary
labels via percentile thresholds, and writes:
  - process_data_summary.csv          (structure-level rows)
  - process_data_summary_atoms.csv    (atom-level rows)
  - process_data_boundaries.json      (threshold values per task)

Usage:
    python scripts/process_data_summary.py \\
        --xyz /path/to/test.xyz \\
        --mace-model-path /path/to/MACE-OFF23_large.model \\
        --output-dir ./probe_data_check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.io_extxyz import iter_probe_extxyz
from probe.labels import (
    atom_force_component_mae,
    structure_mean_force_error,
    compute_percentile_boundary,
)
from probe.backends.mace import (
    load_mace, get_z_table, load_extxyz_dataloader,
    process_batch_mace_multitask,
)


def _labels_from_errors(energy_errors, atom_errors_flat, struct_force_errors,
                        percentile: float):
    boundary_e = compute_percentile_boundary(
        energy_errors, percentile, unit='eV (same as extxyz energy)')
    boundary_f_atom = compute_percentile_boundary(
        atom_errors_flat, percentile, unit='eV/Å (per atom)')
    boundary_f_mol = compute_percentile_boundary(
        struct_force_errors, percentile, unit='eV/Å (structure mean)')

    bins_e = np.array([0.0, boundary_e])
    bins_fa = np.array([0.0, boundary_f_atom])
    bins_fm = np.array([0.0, boundary_f_mol])

    def _bin(err, bins):
        idx = np.digitize(err, bins, right=False) - 1
        return np.clip(idx, 0, len(bins) - 1)

    return {
        'boundary_energy': boundary_e,
        'boundary_force_atom': boundary_f_atom,
        'boundary_force_mol': boundary_f_mol,
        'energy_labels': _bin(energy_errors, bins_e),
        'atom_labels': _bin(atom_errors_flat, bins_fa),
        'struct_force_labels': _bin(struct_force_errors, bins_fm),
    }


def process_file(xyz_path: Path, mace_model_path: Path, output_dir: Path,
                 percentile: float = 50, batch_size: int = 64,
                 device: str = 'cuda', max_structures: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing reference labels from {xyz_path} ...")
    frames = list(iter_probe_extxyz(xyz_path, max_structures=max_structures))
    for sidx, frame in enumerate(frames):
        if frame['true_energy'] is None:
            raise ValueError(f"Structure {sidx}: missing reference energy in info line")

    print(f"Loading MACE model from {mace_model_path} ...")
    extractor = load_mace(str(mace_model_path), device)
    z_table = get_z_table(extractor)
    r_max = float(extractor.mace_model.r_max)

    loader = load_extxyz_dataloader(
        str(xyz_path), z_table, r_max, batch_size=batch_size,
        shuffle=False, max_structures=max_structures)

    struct_rows = []
    atom_rows = []
    energy_errors = []
    atom_errors_all = []
    struct_force_errors = []
    struct_idx = 0

    print("Running MACE forward passes ...")
    for batch in tqdm(loader, desc='MACE inference'):
        (_, atom_mask, pred_e, true_e, pred_f, true_f, _) = \
            process_batch_mace_multitask(batch, device, extractor)

        atom_err = atom_force_component_mae(pred_f, true_f)
        struct_err = structure_mean_force_error(atom_err, atom_mask)
        abs_e = torch.abs(true_e - pred_e)

        B = pred_e.shape[0]
        for i in range(B):
            if struct_idx >= len(frames):
                raise RuntimeError(
                    f"Batch produced more structures than parsed frames "
                    f"({struct_idx + 1} > {len(frames)})")
            frame = frames[struct_idx]
            n = int(atom_mask[i].sum().item())

            e_err = float(abs_e[i].cpu())
            s_err = float(struct_err[i].cpu())
            energy_errors.append(e_err)
            struct_force_errors.append(s_err)

            struct_rows.append({
                'structure_idx': struct_idx,
                'n_atoms': frame['n_atoms'],
                'config_type': frame['config_type'],
                'true_energy': float(true_e[i].cpu()),
                'mace_energy': float(pred_e[i].cpu()),
                'energy_abs_error': e_err,
                'structure_mean_force_mae': s_err,
            })

            atom_err_i = atom_err[i, :n].cpu().numpy()
            pred_f_i = pred_f[i, :n].cpu().numpy()
            true_f_i = true_f[i, :n].cpu().numpy()
            atom_errors_all.extend(atom_err_i.tolist())

            for aidx in range(n):
                tf = true_f_i[aidx]
                mf = pred_f_i[aidx]
                atom_rows.append({
                    'structure_idx': struct_idx,
                    'atom_idx': aidx,
                    'species': frame['symbols'][aidx],
                    'true_fx': tf[0], 'true_fy': tf[1], 'true_fz': tf[2],
                    'mace_fx': mf[0], 'mace_fy': mf[1], 'mace_fz': mf[2],
                    'atom_force_mae': float(atom_err_i[aidx]),
                })
            struct_idx += 1

    if struct_idx != len(frames):
        raise RuntimeError(
            f"MACE dataloader produced {struct_idx} structures, "
            f"expected {len(frames)} (some may have failed conversion)")

    energy_errors = np.asarray(energy_errors)
    atom_errors_all = np.asarray(atom_errors_all)
    struct_force_errors = np.asarray(struct_force_errors)

    print("Computing percentile boundaries and labels ...")
    label_info = _labels_from_errors(
        energy_errors, atom_errors_all, struct_force_errors, percentile)

    struct_df_cols = [
        'structure_idx', 'n_atoms', 'config_type', 'true_energy', 'mace_energy',
        'energy_abs_error', 'energy_label', 'structure_mean_force_mae',
        'structure_force_label', 'boundary_energy', 'boundary_force_atom',
        'boundary_force_mol',
    ]
    atom_df_cols = [
        'structure_idx', 'atom_idx', 'species',
        'true_fx', 'true_fy', 'true_fz', 'mace_fx', 'mace_fy', 'mace_fz',
        'atom_force_mae', 'atom_force_label',
    ]

    struct_path = output_dir / 'process_data_summary.csv'
    atom_path = output_dir / 'process_data_summary_atoms.csv'
    bounds_path = output_dir / 'process_data_boundaries.json'

    energy_labels = label_info['energy_labels']
    struct_force_labels = label_info['struct_force_labels']
    atom_labels_arr = label_info['atom_labels']

    with struct_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=struct_df_cols)
        writer.writeheader()
        for i, row in enumerate(struct_rows):
            writer.writerow({
                **row,
                'energy_label': int(energy_labels[i]),
                'structure_force_label': int(struct_force_labels[i]),
                'boundary_energy': label_info['boundary_energy'],
                'boundary_force_atom': label_info['boundary_force_atom'],
                'boundary_force_mol': label_info['boundary_force_mol'],
            })

    atom_label_idx = 0
    with atom_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=atom_df_cols)
        writer.writeheader()
        for row in atom_rows:
            writer.writerow({
                **row,
                'atom_force_label': int(atom_labels_arr[atom_label_idx]),
            })
            atom_label_idx += 1

    n_struct = len(struct_rows)
    n_atom = len(atom_rows)

    boundaries = {
        'percentile': percentile,
        'n_structures': n_struct,
        'n_atoms': n_atom,
        'mace_model_path': str(mace_model_path),
        'boundary_energy': label_info['boundary_energy'],
        'boundary_force_atom_ev_per_A': label_info['boundary_force_atom'],
        'boundary_force_mol_ev_per_A': label_info['boundary_force_mol'],
        'class_balance': {
            'energy_reliable_frac': float(np.mean(energy_labels == 0)),
            'structure_force_reliable_frac': float(np.mean(struct_force_labels == 0)),
            'atom_force_reliable_frac': float(np.mean(atom_labels_arr == 0)),
        },
        'label_convention': {
            '0': 'reliable (error < boundary)',
            '1': 'unreliable (error >= boundary)',
        },
    }

    bounds_path.write_text(json.dumps(boundaries, indent=2))

    print(f"\nWrote structure summary: {struct_path} ({n_struct} rows)")
    print(f"Wrote atom summary:      {atom_path} ({n_atom} rows)")
    print(f"Wrote boundaries:        {bounds_path}")
    print("\nBoundaries:")
    print(f"  energy:            {label_info['boundary_energy']:.6f} eV")
    print(f"  force (atom):      {label_info['boundary_force_atom']:.6f} eV/Å")
    print(f"  force (structure): {label_info['boundary_force_mol']:.6f} eV/Å")
    print("\nClass balance:")
    for k, v in boundaries['class_balance'].items():
        print(f"  {k}: {v:.3f} reliable fraction")

    if struct_rows:
        s0 = struct_rows[0]
        a0 = atom_rows[0]
        print("\nSanity check — structure 0:")
        print(f"  true_energy={s0['true_energy']}, mace_energy={s0['mace_energy']}, "
              f"err={s0['energy_abs_error']:.6f}, label={int(energy_labels[0])}")
        print(f"  atom0 {a0['species']}: mae={a0['atom_force_mae']:.6f} "
              f"label={int(atom_labels_arr[0])}")


def main():
    parser = argparse.ArgumentParser(
        description='PROBE extxyz labeling summary (live MACE predictions)')
    parser.add_argument('--xyz', required=True, help='Path to extxyz file')
    parser.add_argument('--mace-model-path', required=True,
                        help='Path to frozen MACE model (.model)')
    parser.add_argument('--output-dir', default='./probe_data_check',
                        help='Directory for CSV/JSON outputs')
    parser.add_argument('--percentile', type=float, default=50,
                        help='Error percentile for class boundary (default 50)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default=None,
                        help='cuda or cpu (default: auto)')
    parser.add_argument('--max-structures', type=int, default=None,
                        help='Optional cap for quick tests')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    process_file(
        Path(args.xyz), Path(args.mace_model_path), Path(args.output_dir),
        percentile=args.percentile,
        batch_size=args.batch_size,
        device=device,
        max_structures=args.max_structures,
    )


if __name__ == '__main__':
    main()
