"""Tests for probe.io_extxyz parsing."""

from probe.io_extxyz import read_probe_extxyz_frame


def test_first_frame_matches_raw_test_xyz():
    with open('/Users/zhiyuwang/Desktop/test.xyz') as fh:
        lines = fh.readlines()

    frame, _ = read_probe_extxyz_frame(lines, 0)
    assert frame['n_atoms'] == 19
    assert frame['true_energy'] == -7901.6314939628855
    assert frame['mace_energy'] == -7901.640222079461
    assert abs(frame['true_energy'] - frame['mace_energy'] - 0.008728) < 1e-5

    gt0 = frame['true_forces'][0]
    mace0 = frame['mace_forces'][0]
    assert abs(gt0[0] - (-1.04781723)) < 1e-6
    assert abs(mace0[0] - (-1.05288337)) < 1e-6

    mae0 = abs(gt0 - mace0).mean()
    assert abs(mae0 - 0.009223) < 1e-4
