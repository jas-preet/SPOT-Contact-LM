"""Shared fixtures.

The end-to-end tests run the real command line against tiny scripted models with the
same call signatures, and against tests/fakes in place of ESM-1b.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAKES = REPO_ROOT / "tests" / "fakes"
CONTACT_MODELS = ("atten_single_contact", "atten_only_contact", "atten_all_contact")
DISTANCE_MODELS = ("atten_sgl_dist", "atten_only_dist", "atten_all_dist")
FEATURE_COLUMNS = ("ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta", "Tau",
                   "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B")


class _ContactModel(nn.Module):
    def forward(self, feats):
        return torch.sigmoid(feats.mean(3, keepdim=True))


class _DistanceModel(nn.Module):
    def forward(self, feats):
        bins = feats.mean(3, keepdim=True).repeat(1, 1, 1, 20)
        return bins, bins, bins, bins


@pytest.fixture(scope="session")
def install(tmp_path_factory):
    """A copy of the program with fake checkpoints in the usual location."""
    root = tmp_path_factory.mktemp("install with space")
    shutil.copy(REPO_ROOT / "spot_contact_single.py", root)
    shutil.copytree(REPO_ROOT / "dataset", root / "dataset",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "data_files" / "out_spot_1d_sgl").mkdir(parents=True)
    jits = root / "contact_jits"
    jits.mkdir()
    for name in CONTACT_MODELS:
        torch.jit.script(_ContactModel()).save(str(jits / f"{name}_cpu.pth"))
    for name in DISTANCE_MODELS:
        torch.jit.script(_DistanceModel()).save(str(jits / f"{name}_cpu.pth"))
    return root


@pytest.fixture
def esm_log(tmp_path):
    return tmp_path / "esm.log"


@pytest.fixture
def run_cli(install, esm_log):
    def run(*args, cwd=install):
        env = dict(os.environ, PYTHONPATH=str(FAKES), FAKE_ESM_LOG=str(esm_log))
        return subprocess.run(
            [sys.executable, str(install / "spot_contact_single.py"), *map(str, args)],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )

    return run


def spot_single_table(sequence):
    rng = np.random.default_rng(0)
    table = pd.DataFrame({column: rng.random(len(sequence)) for column in FEATURE_COLUMNS})
    table.insert(0, "AA", list(sequence))
    return table


@pytest.fixture
def write_spot_single(install):
    def write(name, sequence, directory=None):
        directory = directory or install / "data_files" / "out_spot_1d_sgl"
        spot_single_table(sequence).to_csv(directory / f"{name}.csv")

    return write


@pytest.fixture
def make_list(tmp_path):
    """Write FASTA files and a file list that points at them."""

    def make(records, trailer=""):
        paths = []
        for filename, text in records.items():
            path = tmp_path / filename
            path.write_text(text)
            paths.append(str(path))
        file_list = tmp_path / "list.txt"
        file_list.write_text("\n".join(paths) + "\n" + trailer)
        return file_list

    return make
