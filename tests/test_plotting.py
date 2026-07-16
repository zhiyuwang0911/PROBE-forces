"""Smoke test for training-curve plot helper."""

from probe.plotting import update_training_curves


def test_update_training_curves_multitask(tmp_path):
    history = {
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7],
        'train_loss_energy': [0.5, 0.4, 0.3],
        'train_loss_force_atom': [0.4, 0.3, 0.2],
        'train_loss_force_mol': [0.3, 0.2, 0.1],
        'val_acc_energy': [0.5, 0.6, 0.7],
        'val_acc_force_atom': [0.55, 0.65, 0.75],
        'val_acc_force_mol': [0.52, 0.62, 0.72],
    }
    out = update_training_curves(history, tmp_path, best_epoch=2)
    assert out is not None
    assert out.exists()
    assert (tmp_path / 'training_history.json').exists()


def test_update_training_curves_energy_only(tmp_path):
    history = {
        'train_loss': [1.0, 0.5],
        'val_loss': [1.2, 0.6],
        'val_acc': [0.4, 0.8],
    }
    out = update_training_curves(history, tmp_path, best_epoch=2)
    assert out is not None
    assert out.exists()
