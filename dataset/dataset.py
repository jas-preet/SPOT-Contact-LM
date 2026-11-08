import os

import torch
from torch.utils.data import Dataset

from dataset.data_functions import (get_fasta_2d, get_feats_2d, protein_name, read_fasta_file,
                                    read_spot_single)


class ProteinsDataset(Dataset):
    def __init__(self, prot_list, spot_single_dir):
        self.prot_list = prot_list
        self.spot_single_dir = spot_single_dir

    def __len__(self):
        return len(self.prot_list)

    def __getitem__(self, idx):
        fasta_path = self.prot_list[idx]
        prot_name = protein_name(fasta_path)

        seq = read_fasta_file(fasta_path)
        seq_hot = get_fasta_2d(seq)

        spot_sgl_feat = read_spot_single(os.path.join(self.spot_single_dir, prot_name + ".csv"), seq)
        spot_sgl_2d = get_feats_2d(spot_sgl_feat)

        # features
        feature1 = seq_hot
        feature2 = spot_sgl_2d

        return feature1, feature2, prot_name, seq


def protein_collate_fn(data):
    feature1, feature2, prot_name, seq = zip(*data)
    feats_tensor1 = torch.from_numpy(feature1[0])
    feats_tensor2 = torch.from_numpy(feature2[0])

    return feats_tensor1, feats_tensor2, prot_name, seq  ### also return feats_lengths and label_lengths if using packpadd
