#!/usr/bin/env python3
"""
Build labeling summary CSV from a PROBE-style extxyz file.

Reads ground-truth and precomputed MACE energies/forces, computes errors,
assigns binary labels via 50th-percentile thresholds, and writes:
  - process_data_summary.csv          (structure-level rows)
  - process_data_summary_atoms.csv    (atom-level rows)
  - process_data_boundaries.json      (threshold values per task)

Usage:
    python scripts/process_data_summary.py \\
        --xyz /path/to/test.xyz \\
        --output-dir ./probe_data_check
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from probe.io_extxyz import iter_probe_extxyz
from probe.labels import compute_percentile_boundary


def atom_force_mae_numpy(pred_forces: np.ndarray, true_forces: np.ndarray) -> np.ndarray:
    """Per-atom mean absolute force error over x/y/z (numpy)."""
    return np.abs(pred_forces - true_forces).mean(axis=-1)


def _labels_from_errors(energy_errors, atom_errors_flat, struct_force_errors,
                        percentile: float):
    boundary_e = compute_percentile_boundary(
        energy_errors, percentile, unit='(same as file energy unit)')
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


def process_file(xyz_path: Path, output_dir: Path, percentile: float = 50,
                 max_structures: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    struct_rows = []
    atom_rows = []
    energy_errors = []
    atom_errors_all = []
    struct_force_errors = []

    print(f"Reading {xyz_path} ...")
    for sidx, frame in enumerate(tqdm(
            iter_probe_extxyz(xyz_path, max_structures=max_structures),
            desc='Parsing frames')):
        if frame['true_energy'] is None or frame['mace_energy'] is None:
            raise ValueError(
                f"Structure {sidx}: missing energy or MACE_energy in info line")

        true_e = frame['true_energy']
        mace_e = frame['mace_energy']
        e_err = abs(true_e - mace_e)
        energy_errors.append(e_err)

        atom_err = atom_force_mae_numpy(frame['mace_forces'], frame['true_forces'])
        struct_err = float(atom_err.mean())
        struct_force_errors.append(struct_err)
        atom_errors_all.extend(atom_err.tolist())

        struct_rows.append({
            'structure_idx': sidx,
            'n_atoms': frame['n_atoms'],
            'config_type': frame['config_type'],
            'true_energy': true_e,
            'mace_energy': mace_e,
            'energy_abs_error': e_err,
            'structure_mean_force_mae': struct_err,
        })

        for aidx, sym in enumerate(frame['symbols']):
            tf = frame['true_forces'][aidx]
            mf = frame['mace_forces'][aidx]
            atom_rows.append({
                'structure_idx': sidx,
                'atom_idx': aidx,
                'species': sym,
                'true_fx': tf[0], 'true_fy': tf[1], 'true_fz': tf[2],
                'mace_fx': mf[0], 'mace_fy': mf[1], 'mace_fz': mf[2],
                'atom_force_mae': atom_err[aidx],
            })

    energy_errors = np.asarray(energy_errors)
    atom_errors_all = np.asarray(atom_errors_all)
    struct_force_errors = np.asarray(struct_force_errors)

    print("Computing 50th-percentile boundaries and labels ...")
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
    print(f"  energy:           {label_info['boundary_energy']:.6f}")
    print(f"  force (atom):     {label_info['boundary_force_atom']:.6f} eV/Å")
    print(f"  force (structure):{label_info['boundary_force_mol']:.6f} eV/Å")
    print("\nClass balance (expect ~50/50 on this file):")
    for k, v in boundaries['class_balance'].items():
        print(f"  {k}: {v:.3f} reliable fraction")

    # Sanity: first structure manual check
    s0 = struct_rows[0]
    a0 = atom_rows[0]
    e_lbl = int(energy_labels[0])
    sf_lbl = int(struct_force_labels[0])
    af_lbl = int(atom_labels_arr[0])
    print("\nSanity check — structure 0:")
    print(f"  true_energy={s0['true_energy']}, mace_energy={s0['mace_energy']}, "
          f"err={s0['energy_abs_error']:.6f}, label={e_lbl}")
    print(f"  atom0 {a0['species']}: true_f={a0['true_fx']:.4f},{a0['true_fy']:.4f},"
          f"{a0['true_fz']:.4f} mace_f={a0['mace_fx']:.4f},{a0['mace_fy']:.4f},"
          f"{a0['mace_fz']:.4f} mae={a0['atom_force_mae']:.6f} "
          f"label={af_lbl}")
    print(f"  structure_mean_force_mae={s0['structure_mean_force_mae']:.6f}, "
          f"structure_force_label={sf_lbl}")


def main():
    parser = argparse.ArgumentParser(description='PROBE extxyz labeling summary')
    parser.add_argument('--xyz', required=True, help='Path to extxyz file')
    parser.add_argument('--output-dir', default='./probe_data_check',
                        help='Directory for CSV/JSON outputs')
    parser.add_argument('--percentile', type=float, default=50,
                        help='Error percentile for class boundary (default 50)')
    parser.add_argument('--max-structures', type=int, default=None,
                        help='Optional cap for quick tests')
    args = parser.parse_args()

    process_file(
        Path(args.xyz), Path(args.output_dir),
        percentile=args.percentile,
        max_structures=args.max_structures,
    )


if __name__ == '__main__':
    main()
