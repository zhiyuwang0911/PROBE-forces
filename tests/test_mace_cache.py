"""Unit tests for CachedMACEProcessor (no live MACE required)."""

from types import SimpleNamespace

import torch

from probe.backends.mace import CachedMACEProcessor, _pad_cached_structures


def test_pad_cached_structures_force():
    entries = [
        {
            'node_feats': torch.randn(2, 4),
            'pred_energy': torch.tensor(1.0),
            'pred_forces': torch.randn(2, 3),
        },
        {
            'node_feats': torch.randn(3, 4),
            'pred_energy': torch.tensor(2.0),
            'pred_forces': torch.randn(3, 3),
        },
    ]
    true_e = [torch.tensor(1.5), torch.tensor(2.5)]
    true_f = [torch.randn(2, 3), torch.randn(3, 3)]
    out = _pad_cached_structures(entries, true_e, true_f, 'cpu', True)
    atom_feats, atom_mask, pred_e, true_energy, pred_f, true_forces, n_atoms = out
    assert atom_feats.shape == (2, 3, 4)
    assert atom_mask[0].sum() == 2
    assert atom_mask[1].sum() == 3
    assert torch.allclose(pred_e, torch.tensor([1.0, 2.0]))
    assert torch.allclose(n_atoms, torch.tensor([2.0, 3.0]))


def test_cached_processor_hit_after_store(tmp_path):
    """Simulate miss then hit without calling real MACE."""

    class FakeExtractor:
        feat_dim = 4

    proc = CachedMACEProcessor(FakeExtractor(), compute_force=True,
                               cache_dir=str(tmp_path))

    entry = {
        'node_feats': torch.ones(2, 4),
        'pred_energy': torch.tensor(0.5),
        'pred_forces': torch.zeros(2, 3),
    }
    proc._store(7, entry)
    assert proc._load(7) is not None
    assert (tmp_path / '7.pt').exists()

    # Build a fake batch that is fully cached
    batch = SimpleNamespace(
        structure_idx=torch.tensor([7]),
        ptr=torch.tensor([0, 2]),
        energy=torch.tensor([1.0]),
        forces=torch.ones(2, 3),
    )
    out = proc(batch, 'cpu')
    assert proc.hits == 1
    assert out[0].shape == (1, 2, 4)
    assert torch.allclose(out[2], torch.tensor([0.5]))
