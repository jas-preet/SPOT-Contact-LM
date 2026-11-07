"""End-to-end tests of `python spot_contact_single.py ...`."""
import shutil

import numpy as np
import pandas as pd
import pytest

SIMPLE = {"prot.fasta": ">prot\nMKVLA\n"}


@pytest.fixture
def simple_list(make_list, write_spot_single):
    write_spot_single("prot", "MKVLA")
    return make_list(SIMPLE)


def test_writes_a_square_contact_map(run_cli, simple_list, tmp_path):
    out = tmp_path / "out"
    result = run_cli("--file_list", simple_list, "--save_path", out)
    assert result.returncode == 0, result.stderr
    contacts = np.load(out / "prot.npy")
    assert contacts.shape == (5, 5)
    assert ((contacts >= 0) & (contacts <= 1)).all()


def test_writes_only_the_final_map(run_cli, simple_list, tmp_path):
    out = tmp_path / "out"
    run_cli("--file_list", simple_list, "--save_path", out)
    assert [p.name for p in out.iterdir()] == ["prot.npy"]


def test_runs_esm_once_per_protein(run_cli, simple_list, esm_log, tmp_path):
    run_cli("--file_list", simple_list, "--save_path", tmp_path / "out")
    assert len(esm_log.read_text().splitlines()) == 1


def test_save_path_works_without_a_trailing_slash(run_cli, simple_list, tmp_path):
    out = tmp_path / "noslash"
    out.mkdir()
    result = run_cli("--file_list", simple_list, "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.npy").exists()


def test_runs_from_a_different_working_directory(run_cli, simple_list, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "out"
    result = run_cli("--file_list", simple_list, "--save_path", out, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.npy").exists()


def test_jits_dir_option_finds_checkpoints_elsewhere(run_cli, simple_list, install, tmp_path):
    moved = tmp_path / "weights"
    shutil.copytree(install / "contact_jits", moved)
    out = tmp_path / "out"
    result = run_cli("--file_list", simple_list, "--save_path", out, "--jits_dir", moved)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.npy").exists()


def test_spot_single_dir_option_finds_features_elsewhere(run_cli, make_list, write_spot_single, tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    write_spot_single("elsewhere", "MKVLA", directory=features)
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list({"elsewhere.fasta": ">x\nMKVLA\n"}),
                     "--save_path", out, "--spot_single_dir", features)
    assert result.returncode == 0, result.stderr
    assert (out / "elsewhere.npy").exists()


@pytest.mark.parametrize("sequence", ["MKBZUOJXA", "mkvla", "MKVLA*", "MK-LA"])
def test_accepts_unusual_residues(run_cli, make_list, write_spot_single, tmp_path, sequence):
    expected = sequence.rstrip("*").upper()
    write_spot_single("odd", expected)
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list({"odd.fasta": f">odd\n{sequence}\n"}), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert np.load(out / "odd.npy").shape == (len(expected), len(expected))


def test_keeps_dots_in_names(run_cli, make_list, write_spot_single, tmp_path):
    write_spot_single("P1.v1", "MKVLA")
    write_spot_single("P1.v2", "MKVLA")
    records = {"P1.v1.fasta": ">a\nMKVLA\n", "P1.v2.fasta": ">b\nMKVLA\n"}
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list(records), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["P1.v1.npy", "P1.v2.npy"]


def test_ignores_blank_lines_in_file_list(run_cli, make_list, write_spot_single, tmp_path):
    write_spot_single("prot", "MKVLA")
    out = tmp_path / "out"
    result = run_cli("--file_list", make_list(SIMPLE, trailer="\n  \n"), "--save_path", out)
    assert result.returncode == 0, result.stderr
    assert (out / "prot.npy").exists()


def assert_clean_failure(result, message):
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert message in result.stderr


@pytest.mark.parametrize("option", ["--device", "--esm_device"])
def test_rejects_unknown_device(run_cli, simple_list, option):
    assert_clean_failure(run_cli("--file_list", simple_list, option, "banana"), "banana")


def test_requires_file_list(run_cli):
    assert_clean_failure(run_cli(), "--file_list")


def test_reports_missing_checkpoints_with_pointer_to_readme(run_cli, simple_list, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert_clean_failure(run_cli("--file_list", simple_list, "--jits_dir", empty), "README")


def test_reports_missing_spot_single_output(run_cli, make_list):
    result = run_cli("--file_list", make_list({"nofeat.fasta": ">x\nMKVLA\n"}))
    assert_clean_failure(result, "SPOT-1D-Single")


def test_reports_spot_single_output_of_a_different_length(run_cli, make_list, write_spot_single):
    write_spot_single("short", "MKV")
    result = run_cli("--file_list", make_list({"short.fasta": ">x\nMKVLA\n"}))
    assert_clean_failure(result, "short.csv")


def test_reports_spot_single_output_of_a_different_sequence(run_cli, make_list, write_spot_single):
    write_spot_single("other", "MKVLG")
    result = run_cli("--file_list", make_list({"other.fasta": ">x\nMKVLA\n"}))
    assert_clean_failure(result, "other.csv")


def test_rejects_sequences_over_the_esm_limit(run_cli, make_list, write_spot_single):
    sequence = "A" * 1023
    write_spot_single("long", sequence)
    result = run_cli("--file_list", make_list({"long.fasta": f">l\n{sequence}\n"}))
    assert_clean_failure(result, "long.fasta")


def test_rejects_multimer_separators(run_cli, make_list):
    result = run_cli("--file_list", make_list({"dimer.fasta": ">d\nMKV:LAA\n"}))
    assert_clean_failure(result, "dimer.fasta")


def test_rejects_two_files_with_the_same_protein_name(run_cli, write_spot_single, tmp_path):
    write_spot_single("same", "MKVLA")
    paths = []
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        path = tmp_path / folder / "same.fasta"
        path.write_text(">x\nMKVLA\n")
        paths.append(str(path))
    file_list = tmp_path / "dupes.txt"
    file_list.write_text("\n".join(paths) + "\n")
    assert_clean_failure(run_cli("--file_list", file_list), "duplicate")


def test_rejects_names_that_differ_only_by_case(run_cli, make_list, write_spot_single):
    write_spot_single("abc", "MKVLA")
    write_spot_single("ABC", "MKVLA")
    records = {"abc.fasta": ">a\nMKVLA\n", "ABC.fasta": ">b\nMKVLA\n"}
    assert_clean_failure(run_cli("--file_list", make_list(records)), "duplicate")


def test_reports_nan_in_spot_single_output(run_cli, make_list, install, write_spot_single):
    write_spot_single("gap", "MKVLA")
    path = install / "data_files" / "out_spot_1d_sgl" / "gap.csv"
    table = pd.read_csv(path, index_col=0, keep_default_na=False)
    table["Psi"] = table["Psi"].astype(object)
    table.loc[2, "Psi"] = "nan"
    table.to_csv(path)
    assert_clean_failure(run_cli("--file_list", make_list({"gap.fasta": ">x\nMKVLA\n"})), "gap.csv")


def test_reports_unwritable_save_path_before_inference(run_cli, simple_list, esm_log, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result = run_cli("--file_list", simple_list, "--save_path", locked)
    finally:
        locked.chmod(0o700)
    assert_clean_failure(result, "locked")
    assert not esm_log.exists()


def test_refuses_to_exceed_the_memory_limit_on_cpu(run_cli, simple_list, esm_log):
    result = run_cli("--file_list", simple_list, "--max_memory_gb", "0.5")
    assert_clean_failure(result, "--max_memory_gb")
    assert not esm_log.exists()


def test_memory_limit_can_be_raised(run_cli, simple_list, tmp_path):
    out = tmp_path / "out"
    result = run_cli("--file_list", simple_list, "--save_path", out, "--max_memory_gb", "1000")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "text, reason",
    [
        ("MKVLA\n", "header"),
        (">only_a_header\n", "no sequence"),
        (">a\nMKV\n>b\nLAA\n", "one sequence"),
    ],
)
def test_reports_malformed_fasta_by_name(run_cli, make_list, text, reason):
    result = run_cli("--file_list", make_list({"bad.fasta": text}))
    assert_clean_failure(result, "bad.fasta")
    assert reason in result.stderr
