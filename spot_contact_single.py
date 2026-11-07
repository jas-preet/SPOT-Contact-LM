import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.data_functions import (default_memory_limit_gb, estimated_cpu_memory_gb, load_proteins,
                                    protein_name, read_spot_single)
from dataset.dataset import ProteinsDataset, protein_collate_fn

REPO_DIR = Path(__file__).resolve().parent

# ESM-1b has 1024 positions, two of which hold the start and end tokens
ESM_MAX_RESIDUES = 1022
# Letters ESM-1b cannot encode, such as J, are treated as unknown
OUTSIDE_ESM_ALPHABET = re.compile(r"[^LAGVSERTIDPKQNFYMHWCXBUZO]")

# Each checkpoint with the features it takes and whether it predicts distance bins
MODELS = (
    ("atten_single_contact", "last_layer", False),
    ("atten_only_contact", "all_layers", False),
    ("atten_all_contact", "combined", False),
    ("atten_sgl_dist", "last_layer", True),
    ("atten_only_dist", "all_layers", True),
    ("atten_all_dist", "combined", True),
)


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a protein contact map from a single sequence.")
    parser.add_argument('--file_list', required=True, type=str,
                        help='text file listing one FASTA file path per line')
    parser.add_argument('--save_path', default='results/', type=str,
                        help='output directory, created if needed (default: %(default)s)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'],
                        help='the GPU checkpoints were traced on cuda:0 and only run there '
                             '(default: %(default)s)')
    parser.add_argument('--esm_device', default='cpu', type=str,
                        help='device to run ESM-1b on, such as cpu or cuda:0 (default: %(default)s)')
    parser.add_argument('--jits_dir', default=REPO_DIR / 'contact_jits', type=Path,
                        help='directory holding the model checkpoints '
                             '(default: contact_jits/ next to this script)')
    parser.add_argument('--spot_single_dir', default=REPO_DIR / 'data_files' / 'out_spot_1d_sgl',
                        type=Path, help='directory holding SPOT-1D-Single output '
                                        '(default: data_files/out_spot_1d_sgl/ next to this script)')
    parser.add_argument('--max_memory_gb', default=default_memory_limit_gb(), type=float,
                        help='refuse to run on CPU if the estimated memory exceeds this '
                             '(default: three quarters of RAM)')
    return parser.parse_args()


def checkpoint_paths(jits_dir, device):
    suffix = 'cpu' if device == 'cpu' else 'gpu'
    return [Path(jits_dir) / f'{name}_{suffix}.pth' for name, _, _ in MODELS]


def check_device(device):
    try:
        parsed = torch.device(device)
    except RuntimeError:
        sys.exit(f'error: unknown device {device}')
    if parsed.type == 'cuda' and not torch.cuda.is_available():
        sys.exit(f'error: device {device} was requested but CUDA is not available')


def check_spot_single(proteins, spot_single_dir):
    for path, seq in proteins:
        csv_path = Path(spot_single_dir) / f'{protein_name(path)}.csv'
        if not csv_path.is_file():
            sys.exit(f'error: missing SPOT-1D-Single output {csv_path} (see the README)')
        try:
            read_spot_single(csv_path, seq)
        except (KeyError, ValueError) as error:
            sys.exit(f'error: cannot use {csv_path}: {error}')


def check_memory(proteins, limit_gb):
    path, seq = max(proteins, key=lambda protein: len(protein[1]))
    estimate_gb = estimated_cpu_memory_gb(len(seq))
    if estimate_gb > limit_gb:
        sys.exit(f'error: {path} has {len(seq)} residues and needs about {estimate_gb:.0f} GB on CPU, '
                 f'over the {limit_gb:.0f} GB limit. Use a GPU, a shorter sequence, or raise '
                 '--max_memory_gb if you are sure the memory is available')


def check_inputs(args):
    """Report anything that would stop the run before spending time on the models."""
    check_device(args.device)
    check_device(args.esm_device)

    missing = [str(path) for path in checkpoint_paths(args.jits_dir, args.device)
               if not path.is_file()]
    if missing:
        sys.exit('error: missing model checkpoints (see "Installation" in the README):\n  '
                 + '\n  '.join(missing))

    proteins = load_proteins(args.file_list)
    too_long = [path for path, seq in proteins if len(seq) > ESM_MAX_RESIDUES]
    if too_long:
        sys.exit(f'error: ESM-1b accepts up to {ESM_MAX_RESIDUES} residues, these are longer:\n  '
                 + '\n  '.join(too_long))
    check_spot_single(proteins, args.spot_single_dir)
    if args.device == 'cpu':
        check_memory(proteins, args.max_memory_gb)

    try:
        os.makedirs(args.save_path, exist_ok=True)
    except OSError as error:
        sys.exit(f'error: {error}')
    if not os.access(args.save_path, os.W_OK):
        sys.exit(f'error: cannot write to {args.save_path}')
    return [path for path, _ in proteins]


def load_esm(device):
    import esm

    # The ESM-1b checkpoint stores its settings as a Namespace, which PyTorch refuses by default
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    return model.to(device).eval(), alphabet.get_batch_converter()


def load_models(jits_dir, device):
    models = []
    for path in checkpoint_paths(jits_dir, device):
        model = torch.jit.load(str(path)).to(device)
        model.eval()
        models.append(model)
    return models


def esm_attention(esm_model, batch_converter, name, seq, device):
    _, _, batch_tokens = batch_converter([(name, OUTSIDE_ESM_ALPHABET.sub("X", seq))])
    results = esm_model(batch_tokens.to(device), repr_layers=[33], need_head_weights=True)
    return results["attentions"][:, :, :, 1:-1, 1:-1]  # removing start and end tokens


def attention_features(attention, device):
    return apc(symmetrize(attention)).permute(0, 2, 3, 1).to(device, dtype=torch.float)


def contact_probabilities(model, feats, predicts_distance):
    if not predicts_distance:
        return model(feats).squeeze(-1)
    distance_pred, _, _, _ = model(feats)
    distance_pred = F.softmax(distance_pred, dim=3)
    return torch.sum(distance_pred[:, :, :, 1:13], dim=3)


def predict_contacts(models, attention, feat1, feat2, device):
    batch_size, layers, heads, seqlen, _ = attention.size()
    last_layer = attention_features(attention[:, -1], device)
    all_layers = attention_features(attention.reshape(batch_size, layers * heads, seqlen, seqlen), device)
    feats1 = feat1.to(device, dtype=torch.float)[None, :, :, :]
    feats2 = feat2.to(device, dtype=torch.float)[None, :, :, :]
    features = {
        "last_layer": last_layer,
        "all_layers": all_layers,
        "combined": torch.cat((all_layers, feats1, feats2), 3),
    }

    predictions = [contact_probabilities(model, features[kind], predicts_distance).cpu().numpy()
                   for model, (_, kind, predicts_distance) in zip(models, MODELS)]
    return np.mean(np.concatenate(predictions, axis=0), axis=0)


@torch.no_grad()
def main():
    args = parse_args()
    path_list = check_inputs(args)

    dataset = ProteinsDataset(path_list, args.spot_single_dir)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=protein_collate_fn)

    esm_model, batch_converter = load_esm(args.esm_device)
    models = load_models(args.jits_dir, args.device)

    for feat1, feat2, prot_name, seq in tqdm(dataloader):
        attention = esm_attention(esm_model, batch_converter, prot_name[0], seq[0], args.esm_device)
        contacts = predict_contacts(models, attention, feat1, feat2, args.device)
        np.save(os.path.join(args.save_path, prot_name[0] + ".npy"), contacts)


if __name__ == '__main__':
    main()
