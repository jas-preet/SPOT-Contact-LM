import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# Residues are single letters or gaps. A separator such as ":" means several chains.
INVALID_RESIDUE = re.compile(r"[^A-Z-]")

# Maximum accessible surface area per residue, used to convert absolute ASA to relative.
# Anything else (X, B, Z, U, O, J, ...) falls back to UNKNOWN_RESIDUE_ASA.
MAX_ASA = dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
     185, 160, 145, 180, 225, 115, 140, 155, 255, 230),
))
UNKNOWN_RESIDUE_ASA = 1

# Measured on CPU without oneDNN, where the largest models need about 0.145 GB per 1,000 residue
# pairs. ESM-1b and Python hold about 8.5 GB, part of which the largest models can reuse, so this
# estimate is high for long sequences.
CPU_GB_PER_RESIDUE_PAIR = 1.45e-4
CPU_BASE_GB = 8.5


def estimated_cpu_memory_gb(length):
    return CPU_BASE_GB + CPU_GB_PER_RESIDUE_PAIR * length ** 2


def default_memory_limit_gb():
    """Three quarters of the computer's RAM, or no limit if that cannot be determined."""
    try:
        total_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        return float('inf')
    return 0.75 * total_bytes / 2 ** 30


def read_list(file_name):
    """
    returns the list of proteins from a file, ignoring blank lines
    """
    with open(file_name, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def read_fasta_file(fname):
    """
    reads the single sequence from a fasta file
    :param fname: filename (string)
    :return: upper-case protein sequence, without whitespace or a trailing stop "*" (string)
    :raises ValueError: if the file does not hold exactly one valid sequence
    """
    with open(fname, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith('>'):
        raise ValueError(f"{fname}: expected a FASTA header line starting with '>'")
    if any(line.startswith('>') for line in lines[1:]):
        raise ValueError(f"{fname}: expected one sequence per file, found several")

    sequence = ''.join(''.join(lines[1:]).split()).upper().rstrip('*')
    if not sequence:
        raise ValueError(f"{fname}: no sequence after the header")
    invalid = INVALID_RESIDUE.search(sequence)
    if invalid:
        raise ValueError(f"{fname}: unsupported character {invalid.group()!r}, "
                         "only single chains are supported")
    return sequence


def protein_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_proteins(file_list):
    """
    reads every protein in a file list, exiting with a message if any cannot be used
    :param file_list: path of a text file with one fasta path per line (string)
    :return: list of (fasta path, protein sequence)
    """
    try:
        paths = read_list(file_list)
        proteins = [(path, read_fasta_file(path)) for path in paths]
    except (OSError, ValueError) as error:
        sys.exit(f'error: {error}')
    if not proteins:
        sys.exit(f'error: {file_list} does not list any FASTA files')

    # Compared without case, because file names collide that way on macOS and Windows
    names = Counter(protein_name(path).casefold() for path, _ in proteins)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        sys.exit(f'error: duplicate protein names in {file_list}: {", ".join(duplicates)}')
    return proteins


def one_hot(seq):
    RNN_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    return feat


def get_fasta_2d(seq):
    return get_feats_2d(one_hot(seq))


def angle_norm(angle):
    rad_angle = np.deg2rad(angle)
    angle_split = (np.concatenate([np.sin(rad_angle), np.cos(rad_angle)], 1) + 1) / 2.
    return angle_split


def read_spot_single(file_name, seq):
    """
    reads the SPOT-1D-Single output for a protein
    :param file_name: path of the csv file
    :param seq: protein sequence the output must belong to (string)
    :return: features (array)[L,23]
    :raises ValueError: if the output belongs to a different sequence
    """
    data = pd.read_csv(file_name, keep_default_na=False)
    if ''.join(data['AA']).upper() != seq:
        raise ValueError(f"{file_name}: does not match the FASTA sequence, "
                         "regenerate it with SPOT-1D-Single")

    def columns(*names):
        return data[list(names)].to_numpy(dtype=np.float64)

    ss3_prob = columns('P3C', 'P3E', 'P3H').astype(np.float32)
    ss8_prob = columns('P8C', 'P8S', 'P8T', 'P8H', 'P8G', 'P8I', 'P8E', 'P8B').astype(np.float32)

    ASA_den = np.array([MAX_ASA.get(residue, UNKNOWN_RESIDUE_ASA) for residue in seq]).astype(np.float32)[:, None]
    asa_relative = np.clip(columns('ASA') / ASA_den, 0, 1)

    spot_single_feat = np.concatenate(
        (ss3_prob, ss8_prob, asa_relative, columns('HseU'), columns('HseD'), columns('CN'),
         angle_norm(columns('Psi')), angle_norm(columns('Phi')),
         angle_norm(columns('Theta')), angle_norm(columns('Tau'))), 1)
    if not np.isfinite(spot_single_feat).all():
        raise ValueError(f"{file_name}: contains values that are not numbers, "
                         "regenerate it with SPOT-1D-Single")
    return spot_single_feat


def get_feats_2d(feats):
    temp = feats[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    return feature
