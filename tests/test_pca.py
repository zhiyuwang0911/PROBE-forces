import torch
import pytest
from probe.pca import PCAProjection, ModelWithPCA, fit_streaming_pca, restore_model


def test_projection_and_padding():
    projection = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    x = torch.tensor([[[3., 5., 8.], [0., 0., 0.]]])
    mask = torch.tensor([[True, False]])
    assert torch.allclose(projection(x, mask), torch.tensor([[[2., 3.], [0., 0.]]]))


def test_checkpoint_reconstructs_same_coordinates():
    from probe.model import PROBEModel
    p = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    probe = PROBEModel(backbone_dim=2, atom_encoder_output_dim=32,
                       mol_attention_heads=4, dropout=0.0)
    model = ModelWithPCA(probe, p).eval()
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    energy = torch.randn(2)
    y = model(x, mask, energy=energy)
    cfg = dict(backbone_dim=2, atom_encoder_output_dim=32, mol_attention_heads=4,
               dropout=0.0)
    checkpoint = dict(model_config=cfg, pca_dim=2, model_state_dict=model.state_dict())
    restored = restore_model(PROBEModel, checkpoint, 'cpu').eval()
    assert torch.allclose(y, restored(x, mask, energy=energy), atol=1e-6)


def test_fit_train_data_only():
    batch = (torch.tensor([[[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]]),
             torch.tensor([[True, True, True, True]]))
    def processor(b, device):
        return b[0], b[1], None
    projection, info = fit_streaming_pca(processor, [batch], 'cpu', 2, 3)
    assert info['n_training_atoms'] == 4
    assert torch.allclose(projection.mean, torch.tensor([2.5, 3.5, 4.5]))


def test_disabled_pca_preserves_model_checkpoint():
    from probe.model import PROBEModel
    from probe.pca import make_pca
    projection, info = make_pca(None, None, 'cpu', None, 3)
    assert projection is None and info is None
    cfg = dict(backbone_dim=3, atom_encoder_output_dim=32, mol_attention_heads=4, dropout=0.)
    model = PROBEModel(**cfg).eval()
    restored = restore_model(PROBEModel, dict(model_config=cfg, pca_dim=None,
                             model_state_dict=model.state_dict()), 'cpu').eval()
    x, mask = torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(model(x, mask), restored(x, mask))


def test_invalid_pca_options():
    import pytest
    from probe.pca import make_pca
    for dim in (0, -1, 4):
        with pytest.raises(ValueError):
            make_pca(None, None, 'cpu', dim, 3)


@pytest.mark.parametrize('dim', [None, 2])
def test_multitask_training_resume_inference(monkeypatch, tmp_path, dim):
    import sys
    from types import SimpleNamespace
    import train_mace_multitask as training
    import infer_mace_multitask as inference
    import probe.train as loops
    torch.manual_seed(7)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    data = (torch.randn(2, 3, 4), mask, torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.25]), torch.randn(2, 3, 3), torch.randn(2, 3, 3), mask.sum(1))
    batch = SimpleNamespace(ptr=torch.tensor([0, 2, 5]))
    loader = [batch]
    extractor = SimpleNamespace(feat_dim=4, mace_model=SimpleNamespace(r_max=4.))
    monkeypatch.setattr(training, 'load_mace', lambda *a, **k: extractor)
    monkeypatch.setattr(training, 'get_z_table', lambda *a: None)
    monkeypatch.setattr(training, 'train_val_split_loader', lambda *a, **k: (loader, loader))
    class Processor:
        def __init__(self, *a, **k):
            self.hits = self.misses = 0
        def __call__(self, *a):
            return data
        def __len__(self):
            return 2
    monkeypatch.setattr(training, 'CachedMACEProcessor', Processor)
    monkeypatch.setattr(training, 'scan_force_error_boundaries', lambda *a, **k: (.5, .5))
    monkeypatch.setattr(loops, 'update_training_curves', lambda *a, **k: None)
    for k,v in dict(device='cpu', output_dir=str(tmp_path/'train'), epochs=1,
                    atom_encoder_hidden=[16], atom_encoder_output_dim=16,
                    mol_attention_heads=4, classifier_hidden=[8],
                    atom_force_head_hidden=[8], dropout=0., high_conf_cutoffs=None).items():
        monkeypatch.setitem(training.CONFIG, k, v)
    monkeypatch.setattr(sys, 'argv', ['train', '--checkpoint-every', '1'] + (['--pca-dim', str(dim)] if dim else []))
    training.main()
    last = tmp_path/'train'/'last_checkpoint.pt'
    before = torch.load(last, weights_only=False)
    best = list((tmp_path/'train').glob('best_multitask_model_*.pt'))[0]
    for path in [best, tmp_path/'train'/'checkpoint_epoch_0001.pt']:
        ckpt = torch.load(path, weights_only=False)
        assert ckpt['pca_dim'] == dim and ckpt['original_dim'] == 4
        assert ckpt['model_config']['backbone_dim'] == (dim or 4)
    if dim is None:
        # Simulate a legacy full-dimensional checkpoint with no architecture/PCA metadata.
        legacy = {k:v for k,v in before.items() if k not in ('model_config', 'pca_dim', 'original_dim', 'pca_info', 'task')}
        torch.save(legacy, last)
    def no_refit(*a, **k):
        raise AssertionError('Resume must not refit PCA')
    monkeypatch.setattr(training, 'make_pca', no_refit)
    monkeypatch.setitem(training.CONFIG, 'epochs', 2)
    monkeypatch.setattr(sys, 'argv', ['train', '--resume'])
    training.main()
    after = torch.load(last, weights_only=False)
    assert after['epoch'] == 2
    if dim is not None:
        for key in ('pca.mean', 'pca.components'):
            assert torch.equal(before['model_state_dict'][key], after['model_state_dict'][key])
    from probe.model import MultitaskPROBEModel
    from probe.pca import restore_model
    model = restore_model(MultitaskPROBEModel, after, 'cpu').eval()
    monkeypatch.setattr(inference, 'process_batch_mace_multitask', lambda *a: data)
    bins = torch.tensor([0., .5])
    rows, atom_rows, metrics = inference.run_inference(model, extractor, loader, 'cpu', bins, bins, bins, True)
    assert len(rows) == 2 and len(atom_rows) == 5
    assert set(metrics) == {'energy', 'force_mol', 'force_atom'}
    monkeypatch.setattr(inference, 'load_mace', lambda *a, **k: extractor)
    monkeypatch.setattr(inference, 'get_z_table', lambda *a: None)
    monkeypatch.setattr(inference, 'load_extxyz_dataloader', lambda *a, **k: loader)
    monkeypatch.setattr(sys, 'argv', ['infer', '--mace-model', 'fake', '--checkpoint', str(last),
                                    '--test-xyz', 'fake', '--device', 'cpu', '--output-dir', str(tmp_path/'infer')])
    inference.main()
    assert (tmp_path/'infer'/'predictions_structure.csv').exists()
    monkeypatch.setattr(sys, 'argv', ['train', '--resume', '--pca-dim', '3'])
    import pytest
    with pytest.raises(ValueError, match='unchanged PCA dimension'):
        training.main()
