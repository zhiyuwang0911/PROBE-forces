"""Tests for probe.io_extxyz parsing (reference labels only)."""

from probe.io_extxyz import read_probe_extxyz_frame


SAMPLE_FRAME = """\
3
energy=-10.0 MACE_energy=-10.5 config_type="test"
C 0.0 0.0 0.0 0.1 0.2 0.3 0.4 0.5 0.6
H 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
H 0.0 1.0 0.0 -0.1 -0.2 -0.3
"""


def test_parses_reference_energy_and_forces_ignores_mace_columns():
    lines = SAMPLE_FRAME.splitlines(keepends=True)
    frame, next_idx = read_probe_extxyz_frame(lines, 0)

    assert frame['n_atoms'] == 3
    assert frame['true_energy'] == -10.0
    assert 'mace_energy' not in frame
    assert 'mace_forces' not in frame
    assert next_idx == len(lines)

    assert frame['true_forces'][0].tolist() == [0.1, 0.2, 0.3]
    assert frame['true_forces'][1].tolist() == [0.0, 0.0, 0.0]
    assert frame['true_forces'][2].tolist() == [-0.1, -0.2, -0.3]


def test_first_frame_from_user_xyz_if_present():
    """Optional integration check against local test.xyz (reference fields only)."""
    path = '/Users/zhiyuwang/Desktop/test.xyz'
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return

    frame, _ = read_probe_extxyz_frame(lines, 0)
    assert frame['n_atoms'] == 19
    assert frame['true_energy'] == -7901.6314939628855
    assert 'mace_energy' not in frame

    gt0 = frame['true_forces'][0]
    assert abs(gt0[0] - (-1.04781723)) < 1e-6
