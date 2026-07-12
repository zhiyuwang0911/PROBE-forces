#!/usr/bin/env python3
"""
Inference for multitask PROBE (energy + force) on MACE.

Requires:
  - MACE .model
  - best_multitask_model_*.pt from train_mace_multitask.py
  - test.xyz / .extxyz (reference energy/forces needed for metrics)

Example:
  python infer_mace_multitask.py \\
    --mace-model /path/to/MACE-OFF23_large.model \\
    --checkpoint /path/to/best_multitask_model_YYYYMMDD_HHMMSS.pt \\
    --test-xyz /path/to/test.xyz \\
    --output-dir ./probe_multitask_inference
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from probe.model import MultitaskPROBEModel
from probe.backends.mace import (
    load_mace,
    get_z_table,
    load_extxyz_dataloader,
    process_batch_mace_multitask,
)
from probe.labels import (
    atom_force_component_mae,
    structure_mean_force_error,
    scalar_to_bin_index,
)
from probe.metrics import compute_all_metrics, confusion_matrix_torch


def parse_args():
    p = argparse.ArgumentParser(description="Multitask PROBE inference (E + Fa + Fs)")
    p.add_argument("--mace-model", required=True)
    p.add_argument("--checkpoint", required=True, help="best_multitask_model_*.pt")
    p.add_argument("--test-xyz", required=True)
    p.add_argument("--output-dir", default="./probe_multitask_inference")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--enable-cueq", action="store_true")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--atom-encoder-hidden", nargs="+", type=int, default=[256, 128])
    p.add_argument("--atom-encoder-output-dim", type=int, default=256)
    p.add_argument("--mol-attention-heads", type=int, default=32)
    p.add_argument("--classifier-hidden", nargs="+", type=int, default=[256, 128, 32])
    p.add_argument("--atom-force-head-hidden", nargs="+", type=int, default=[128, 32])
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def _bins(ckpt, key, device):
    if key not in ckpt:
        raise KeyError(f"Checkpoint missing {key!r}")
    return torch.tensor(ckpt[key], device=device, dtype=torch.float32)


@torch.no_grad()
def run_inference(model, extractor, loader, device,
                  bins_e, bins_fa, bins_fm, compute_metrics: bool):
    model.eval()
    struct_rows, atom_rows = [], []
    store = {
        "energy": {"logits": [], "targets": []},
        "force_mol": {"logits": [], "targets": []},
        "force_atom": {"logits": [], "targets": []},
    }
    next_struct = 0

    for batch in tqdm(loader, desc="Inference"):
        B = batch.ptr.shape[0] - 1
        (atom_feats, atom_mask, pred_e, true_e,
         pred_f, true_f, n_atoms) = process_batch_mace_multitask(
            batch, device, extractor)

        valid = ~torch.isnan(pred_e) & torch.isfinite(pred_f).all(dim=(1, 2))

        for b in range(B):
            global_i = next_struct + b
            if not bool(valid[b].item()):
                continue

            af = atom_feats[b:b + 1]
            am = atom_mask[b:b + 1]
            pe = pred_e[b:b + 1]
            te = true_e[b:b + 1]
            pf = pred_f[b:b + 1]
            tf = true_f[b:b + 1]
            na = n_atoms[b:b + 1]

            logits_e, logits_fa, logits_fm = model(af, am, energy=pe)
            probs_e = F.softmax(logits_e, dim=-1)[0]
            probs_fm = F.softmax(logits_fm, dim=-1)[0]
            probs_fa = F.softmax(logits_fa, dim=-1)[0]

            abs_e = torch.abs(te - pe)[0]
            atom_err = atom_force_component_mae(pf, tf)[0]
            struct_err = structure_mean_force_error(atom_err.unsqueeze(0), am)[0]

            struct_rows.append({
                "structure_index": global_i,
                "n_atoms": int(na[0].item()),
                "pred_energy_eV": float(pe[0].item()),
                "true_energy_eV": float(te[0].item()) if torch.isfinite(te[0]) else float("nan"),
                "abs_energy_error_eV": float(abs_e.item()) if torch.isfinite(abs_e) else float("nan"),
                "mean_force_error_eV_A": float(struct_err.item()) if torch.isfinite(struct_err) else float("nan"),
                "p_unreliable_energy": float(probs_e[1].item()),
                "pred_energy": int(probs_e.argmax().item()),
                "p_unreliable_force_mol": float(probs_fm[1].item()),
                "pred_force_mol": int(probs_fm.argmax().item()),
            })

            n = int(am[0].sum().item())
            for a in range(n):
                atom_rows.append({
                    "structure_index": global_i,
                    "atom_index": a,
                    "force_error_eV_A": float(atom_err[a].item()),
                    "p_unreliable_force_atom": float(probs_fa[a, 1].item()),
                    "pred_force_atom": int(probs_fa[a].argmax().item()),
                })

            if compute_metrics:
                store["energy"]["logits"].append(logits_e.cpu())
                store["energy"]["targets"].append(
                    scalar_to_bin_index(abs_e.unsqueeze(0), bins_e).cpu())
                store["force_mol"]["logits"].append(logits_fm.cpu())
                store["force_mol"]["targets"].append(
                    scalar_to_bin_index(struct_err.unsqueeze(0), bins_fm).cpu())
                store["force_atom"]["logits"].append(logits_fa.cpu()[0, :n])
                store["force_atom"]["targets"].append(
                    scalar_to_bin_index(atom_err[:n], bins_fa).cpu())

        next_struct += B

    metrics = {}
    if compute_metrics and store["energy"]["logits"]:
        for task in ("energy", "force_mol", "force_atom"):
            logits = torch.cat(store[task]["logits"], dim=0)
            targets = torch.cat(store[task]["targets"], dim=0)
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            cm = confusion_matrix_torch(preds, targets, n_classes=2)
            metrics[task] = compute_all_metrics(cm)
            metrics[task]["n"] = int(len(targets))
    return struct_rows, atom_rows, metrics


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    extractor = load_mace(args.mace_model, device, enable_cueq=args.enable_cueq)
    z_table = get_z_table(extractor)
    r_max = float(extractor.mace_model.r_max)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")
    bins_e = _bins(ckpt, "error_bins_energy", device)
    bins_fa = _bins(ckpt, "error_bins_force_atom", device)
    bins_fm = _bins(ckpt, "error_bins_force_mol", device)
    print(f"error_bins energy={bins_e.tolist()}")
    print(f"error_bins force_atom={bins_fa.tolist()}")
    print(f"error_bins force_mol={bins_fm.tolist()}")

    model = MultitaskPROBEModel(
        backbone_dim=extractor.feat_dim,
        atom_encoder_hidden=args.atom_encoder_hidden,
        atom_encoder_output_dim=args.atom_encoder_output_dim,
        mol_attention_heads=args.mol_attention_heads,
        classifier_hidden=args.classifier_hidden,
        atom_force_head_hidden=args.atom_force_head_hidden,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = load_extxyz_dataloader(
        args.test_xyz, z_table, r_max,
        batch_size=args.batch_size, shuffle=False,
        max_structures=args.max_structures,
    )
    print(f"Batches: {len(loader)}")

    struct_rows, atom_rows, metrics = run_inference(
        model, extractor, loader, device, bins_e, bins_fa, bins_fm,
        not args.no_metrics,
    )

    struct_fields = [
        "structure_index", "n_atoms", "pred_energy_eV", "true_energy_eV",
        "abs_energy_error_eV", "mean_force_error_eV_A",
        "p_unreliable_energy", "pred_energy",
        "p_unreliable_force_mol", "pred_force_mol",
    ]
    atom_fields = [
        "structure_index", "atom_index", "force_error_eV_A",
        "p_unreliable_force_atom", "pred_force_atom",
    ]

    with open(out_dir / "predictions_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=struct_fields)
        w.writeheader()
        w.writerows(struct_rows)
    print(f"Wrote {len(struct_rows)} structures → {out_dir / 'predictions_structure.csv'}")

    with open(out_dir / "predictions_atom.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=atom_fields)
        w.writeheader()
        w.writerows(atom_rows)
    print(f"Wrote {len(atom_rows)} atoms → {out_dir / 'predictions_atom.csv'}")

    np.savez_compressed(
        out_dir / "predictions.npz",
        structure_index=np.array([r["structure_index"] for r in struct_rows], dtype=np.int64),
        p_unreliable_energy=np.array([r["p_unreliable_energy"] for r in struct_rows], dtype=np.float32),
        pred_energy=np.array([r["pred_energy"] for r in struct_rows], dtype=np.int64),
        p_unreliable_force_mol=np.array([r["p_unreliable_force_mol"] for r in struct_rows], dtype=np.float32),
        pred_force_mol=np.array([r["pred_force_mol"] for r in struct_rows], dtype=np.int64),
        atom_structure_index=np.array([r["structure_index"] for r in atom_rows], dtype=np.int64),
        atom_index=np.array([r["atom_index"] for r in atom_rows], dtype=np.int64),
        p_unreliable_force_atom=np.array([r["p_unreliable_force_atom"] for r in atom_rows], dtype=np.float32),
        pred_force_atom=np.array([r["pred_force_atom"] for r in atom_rows], dtype=np.int64),
    )

    if metrics:
        for task, m in metrics.items():
            print(f"{task}: acc={m['accuracy']:.4f} mcc={m['mcc']:.4f} f1={m['f1']:.4f} n={m['n']}")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "test_xyz": args.test_xyz,
            "n_structures": len(struct_rows),
            "n_atoms": len(atom_rows),
            "error_bins_energy": bins_e.cpu().tolist(),
            "error_bins_force_atom": bins_fa.cpu().tolist(),
            "error_bins_force_mol": bins_fm.cpu().tolist(),
            "label_meaning": {"0": "reliable", "1": "unreliable"},
        }, f, indent=2)
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
