import numpy as np
import pytest

from conftest import spot_single_table
from dataset.data_functions import (angle_norm, estimated_cpu_memory_gb, get_feats_2d, one_hot, read_fasta_file, read_list,
                                    read_spot_single)


def test_one_hot_marks_the_matching_residue():
    encoded = one_hot("AV")
    assert encoded.shape == (2, 20)
    assert encoded[0, 0] == 1 and encoded[0].sum() == 1
    assert encoded[1, 19] == 1 and encoded[1].sum() == 1


def test_read_list_skips_blank_lines(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("a.fasta\n\n  b.fasta  \n   \n")
    assert read_list(path) == ["a.fasta", "b.fasta"]


def test_read_fasta_joins_wrapped_lines_and_ignores_whitespace(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text(">x description\nMKV \n\nLA\r\n\n")
    assert read_fasta_file(path) == "MKVLA"


def test_angle_norm_scales_sin_and_cos_to_zero_one():
    assert angle_norm(np.array([[90.0]])) == pytest.approx(np.array([[1.0, 0.5]]))


def test_feats_2d_pairs_every_residue_with_every_other():
    feats = np.array([[1.0], [2.0]])
    assert get_feats_2d(feats).tolist() == [[[1.0, 1.0], [2.0, 1.0]], [[1.0, 2.0], [2.0, 2.0]]]


def write_table(tmp_path, sequence):
    path = tmp_path / "x.csv"
    spot_single_table(sequence).to_csv(path)
    return path


def test_read_spot_single_returns_all_features(tmp_path):
    assert read_spot_single(write_table(tmp_path, "MKV"), "MKV").shape == (3, 23)


def test_read_spot_single_scales_asa_by_each_residues_maximum(tmp_path):
    path = write_table(tmp_path, "A")
    table = spot_single_table("A")
    table["ASA"] = 57.5
    table.to_csv(path)
    assert read_spot_single(path, "A")[0, 11] == pytest.approx(0.5)


def test_read_spot_single_accepts_nonstandard_residues(tmp_path):
    assert read_spot_single(write_table(tmp_path, "BJ"), "BJ").shape == (2, 23)


@pytest.mark.parametrize("length, measured_model_gb", [(150, 3.45), (250, 8.93), (300, 12.75)])
def test_memory_estimate_covers_measured_model_peaks(length, measured_model_gb):
    assert measured_model_gb + 8 <= estimated_cpu_memory_gb(length) < measured_model_gb + 10
