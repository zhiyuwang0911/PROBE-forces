"""
MACE backend for PROBE.

Usage:
    from probe.backends.mace import load_mace, process_batch_mace, MACEFeatureExtractor

    extractor = load_mace(model_path, device)
    model     = PROBEModel(backbone_dim=extractor.feat_dim)

    # In training loop:
    process_fn = lambda batch, dev: process_batch_mace(batch, dev, extractor)
    run_training(model, process_fn, ...)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# e3nn and MACE must be installed / on sys.path
try:
    from e3nn import o3
    from mace.tools import AtomicNumberTable, atomic_numbers_to_indices
    from mace.tools import torch_geometric
    from mace.data.atomic_data import AtomicData
    from mace.data.utils import Configuration
    import ase.io
    import ase.data
    _MACE_AVAILABLE = True
except ImportError:
    _MACE_AVAILABLE = False

try:
    from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq
    _CUEQ_AVAILABLE = True
except ImportError:
    _CUEQ_AVAILABLE = False
    convert_e3nn_to_cueq = None


# ---------------------------------------------------------------------------
# Feature extractor (forward hook)
# ---------------------------------------------------------------------------

class MACEFeatureExtractor(nn.Module):
    """Wraps a MACE model and captures scalar node features from the last
    product block via a forward hook. The backbone is never modified."""

    def __init__(self, mace_model):
        if not _MACE_AVAILABLE:
            raise ImportError("MACE is not installed. "
                              "See https://github.com/ACEsuit/mace")
        super().__init__()
        self.mace_model = mace_model
        self._last_feats = None

        last_product = self.mace_model.products[-1]
        last_product.register_forward_hook(self._hook)

        # Auto-detect scalar feature dimension (L=0 irreps)
        self.feat_dim = last_product.linear.irreps_out.count(o3.Irrep(0, 1))
        print(f"MACEFeatureExtractor: feat_dim={self.feat_dim} "
              f"(from products[-1])")

    def _hook(self, module, input, output):
        self._last_feats = output

    def forward(self, data, compute_force: bool = False):
        """Returns (mace_output_dict, node_feats [n_atoms, feat_dim]).

        Force prediction uses autograd w.r.t. positions and must run outside
        torch.no_grad(). Backbone weights stay frozen (requires_grad=False).
        """
        self._last_feats = None
        if compute_force:
            # enable_grad() overrides an outer torch.no_grad() (e.g. validation).
            with torch.enable_grad():
                if hasattr(data, 'positions') and data.positions is not None:
                    data.positions = data.positions.detach().requires_grad_(True)
                out = self.mace_model(data, compute_force=True)
        else:
            with torch.no_grad():
                out = self.mace_model(data, compute_force=False)
        assert self._last_feats is not None, "Forward hook did not fire."
        return out, self._last_feats


# ---------------------------------------------------------------------------
# Load backbone
# ---------------------------------------------------------------------------

def load_mace(model_path: str, device: str = 'cuda',
              enable_cueq: bool = False) -> MACEFeatureExtractor:
    """Load a frozen MACE model and return a MACEFeatureExtractor.

    Args:
        model_path: Path to a MACE .model checkpoint.
        device: 'cuda' or 'cpu'.
        enable_cueq: If True, convert the model to NVIDIA cuEquivariance
            CUDA kernels (requires cuequivariance packages and CUDA).
    """
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    mace_model = torch.load(model_path, map_location=device)

    if enable_cueq:
        if not str(device).startswith('cuda'):
            raise ValueError(
                "enable_cueq=True requires a CUDA device "
                f"(got device={device!r}).")
        if not _CUEQ_AVAILABLE:
            raise ImportError(
                "enable_cueq=True but cuEquivariance is not installed. "
                "Install cuequivariance, cuequivariance-torch, and "
                "cuequivariance-ops-torch-cu12 — see "
                "https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html"
            )
        print("Converting MACE model to cuEquivariance (CUDA acceleration)...")
        mace_model = convert_e3nn_to_cueq(
            mace_model, device=device, return_model=True)

    mace_model.to(device).eval().float()   # cast to float32
    for p in mace_model.parameters():
        p.requires_grad = False
    extractor = MACEFeatureExtractor(mace_model)
    cueq_status = 'on' if enable_cueq else 'off'
    print(f"r_max={mace_model.r_max:.2f}, "
          f"num_interactions={mace_model.num_interactions.item()}, "
          f"feat_dim={extractor.feat_dim}, cueq={cueq_status}")
    return extractor


def get_z_table(mace_extractor: MACEFeatureExtractor) -> 'AtomicNumberTable':
    zs = mace_extractor.mace_model.atomic_numbers.cpu().tolist()
    return AtomicNumberTable(sorted(zs))


# ---------------------------------------------------------------------------
# Data loading (XYZ → PyG DataLoader)
# ---------------------------------------------------------------------------

def _get_energy(atoms):
    for key in ('energy', 'REF_energy', 'DFT_energy'):
        val = atoms.info.get(key)
        if val is not None:
            return float(val)
    if atoms.calc is not None and 'energy' in getattr(atoms.calc, 'results', {}):
        return float(atoms.calc.results['energy'])
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def frame_dict_to_atoms(frame: dict):
    """Convert a probe.io_extxyz frame dict to an ASE Atoms object."""
    from ase import Atoms
    atoms = Atoms(symbols=frame['symbols'], positions=frame['positions'])
    if frame['true_energy'] is not None:
        atoms.info['energy'] = frame['true_energy']
    if frame.get('config_type'):
        atoms.info['config_type'] = frame['config_type']
    atoms.arrays['forces'] = frame['true_forces']
    return atoms


def load_probe_extxyz_list(xyz_path: str, max_structures: int = None):
    """Load structures with reference energy/forces via probe.io_extxyz."""
    from ..io_extxyz import iter_probe_extxyz
    atoms_list = []
    for frame in iter_probe_extxyz(xyz_path, max_structures=max_structures):
        atoms_list.append(frame_dict_to_atoms(frame))
    return atoms_list


def atoms_to_atomic_data(atoms, z_table, r_max, heads=None):
    if heads is None:
        heads = ['Default']
    energy = _get_energy(atoms) or 0.0
    config = Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        properties={
            'energy': energy,
            'forces': atoms.arrays.get('forces', np.zeros((len(atoms), 3))),
        },
        property_weights={'energy': 1.0, 'forces': 1.0},
        pbc=tuple(atoms.get_pbc().tolist()),
        cell=np.array(atoms.get_cell()),
        config_type=atoms.info.get('config_type', 'Default'),
        head='Default',
    )
    return AtomicData.from_config(config, z_table=z_table,
                                  cutoff=r_max, heads=heads)


def load_xyz_dataloader(xyz_path: str, z_table, r_max: float,
                        batch_size: int, shuffle: bool = True,
                        max_structures: int = None):
    """Load an XYZ file into a PyG DataLoader."""
    from tqdm.auto import tqdm
    atoms_list = ase.io.read(xyz_path, index=':')
    if max_structures:
        atoms_list = atoms_list[:max_structures]
    dataset = []
    for atoms in tqdm(atoms_list, desc='Converting', leave=False):
        try:
            dataset.append(atoms_to_atomic_data(atoms, z_table, r_max))
        except Exception:
            pass
    return torch_geometric.DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)


def train_val_split_loader(xyz_path: str, z_table, r_max: float,
                           batch_size: int, valid_fraction: float = 0.1,
                           seed: int = 42, use_probe_extxyz: bool = True):
    """Load XYZ and return (train_loader, val_loader).

    Uses probe.io_extxyz when ASE cannot parse reference energy/forces
  from complex extxyz info lines. MACE predictions are always computed live.
    """
    import numpy as np
    from tqdm.auto import tqdm

    if use_probe_extxyz:
        atoms_list = load_probe_extxyz_list(xyz_path)
    else:
        atoms_list = ase.io.read(xyz_path, index=':')

    rng = np.random.default_rng(seed)
    idx = np.arange(len(atoms_list))
    rng.shuffle(idx)
    n_val = max(1, int(len(atoms_list) * valid_fraction))
    val_set = set(idx[:n_val].tolist())
    train_data, val_data = [], []
    for i, atoms in enumerate(tqdm(atoms_list, desc='Converting', leave=False)):
        try:
            d = atoms_to_atomic_data(atoms, z_table, r_max)
            (val_data if i in val_set else train_data).append(d)
        except Exception:
            pass
    train_loader = torch_geometric.DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True,  drop_last=False)
    val_loader   = torch_geometric.DataLoader(val_data,   batch_size=batch_size,
                                              shuffle=False, drop_last=False)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    return train_loader, val_loader


def load_extxyz_dataloader(xyz_path: str, z_table, r_max: float,
                           batch_size: int, shuffle: bool = False,
                           max_structures: int = None):
    """Load extxyz (reference labels) into a PyG DataLoader."""
    from tqdm.auto import tqdm
    atoms_list = load_probe_extxyz_list(xyz_path, max_structures=max_structures)
    dataset = []
    for atoms in tqdm(atoms_list, desc='Converting', leave=False):
        try:
            dataset.append(atoms_to_atomic_data(atoms, z_table, r_max))
        except Exception:
            pass
    return torch_geometric.DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
# Batch processing  (PyG-style flat atom tensors → [B, N_max, D] padded)
# ---------------------------------------------------------------------------

def process_batch_mace(batch, device: str, extractor: MACEFeatureExtractor,
                       compute_force: bool = False):
    """
    Run MACE on a PyG batch and return PROBE-compatible padded tensors.

    Predicted energy/forces come from the live MACE forward pass.
    Reference energy/forces come from the dataset (extxyz).

    Returns:
        atom_feats:   [B, N_max, feat_dim]
        atom_mask:    [B, N_max] bool
        pred_energy:  [B]
        true_energy:  [B]
        n_atoms:      [B]
        pred_forces:  [B, N_max, 3]  (only if compute_force=True)
        true_forces:  [B, N_max, 3]  (only if compute_force=True)
    """
    batch = batch.to(device)
    for key in batch.keys:
        attr = getattr(batch, key, None)
        if isinstance(attr, torch.Tensor) and attr.is_floating_point():
            setattr(batch, key, attr.float())

    mace_out, node_feats_flat = extractor(batch, compute_force=compute_force)
    node_feats_flat = node_feats_flat.detach()

    ptr         = batch.ptr
    pred_energy = mace_out['energy'].detach()
    true_energy = batch.energy
    B           = ptr.shape[0] - 1
    D           = node_feats_flat.shape[1]
    sizes       = (ptr[1:] - ptr[:-1]).tolist()
    N_max       = max(sizes)

    atom_feats = torch.zeros(B, N_max, D, device=device)
    atom_mask  = torch.zeros(B, N_max, dtype=torch.bool, device=device)
    pred_forces = true_forces = None
    if compute_force:
        pred_forces_flat = mace_out['forces'].detach()
        true_forces_flat = batch.forces
        pred_forces = torch.zeros(B, N_max, 3, device=device)
        true_forces = torch.zeros(B, N_max, 3, device=device)

    for i in range(B):
        s, e = ptr[i].item(), ptr[i + 1].item()
        n = e - s
        atom_feats[i, :n] = node_feats_flat[s:e]
        atom_mask[i, :n]  = True
        if compute_force:
            pred_forces[i, :n] = pred_forces_flat[s:e]
            true_forces[i, :n] = true_forces_flat[s:e]

    n_atoms = atom_mask.sum(dim=1).float()
    if compute_force:
        return (atom_feats, atom_mask, pred_energy, true_energy,
                pred_forces, true_forces, n_atoms)
    return atom_feats, atom_mask, pred_energy, true_energy, n_atoms


def process_batch_mace_multitask(batch, device: str,
                                 extractor: MACEFeatureExtractor):
    """Multitask batch processing: live MACE energy + forces."""
    return process_batch_mace(batch, device, extractor, compute_force=True)


def scan_force_error_boundaries(train_loader, device, extractor,
                                percentile: float = 50):
    """Scan training set and return (boundary_atom, boundary_mol) in eV/Å."""
    from ..labels import (atom_force_component_mae, structure_mean_force_error,
                          compute_percentile_boundary)

    atom_errors, struct_errors = [], []
    for batch in train_loader:
        (_, atom_mask, _, _, pred_forces, true_forces, _) = \
            process_batch_mace_multitask(batch, device, extractor)
        atom_err = atom_force_component_mae(pred_forces, true_forces)
        struct_err = structure_mean_force_error(atom_err, atom_mask)
        mask = atom_mask.bool()
        atom_errors.extend(atom_err[mask].cpu().numpy().tolist())
        struct_errors.extend(struct_err.cpu().numpy().tolist())

    boundary_atom = compute_percentile_boundary(
        np.array(atom_errors), percentile, unit='eV/Å (per atom)')
    boundary_mol = compute_percentile_boundary(
        np.array(struct_errors), percentile, unit='eV/Å (per structure mean)')
    return boundary_atom, boundary_mol
