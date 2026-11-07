import os

import torch
from torch import nn

# The residues ESM-1b can encode. Any other letter raises KeyError in the real batch converter.
ALPHABET = "LAGVSERTIDPKQNFYMHWCXBUZO.-"


class FakeEsm(nn.Module):
    def forward(self, tokens, repr_layers=(), need_head_weights=False, return_contacts=False):
        assert not self.training, "ESM must run in eval mode"
        assert not return_contacts, "the contact regression is unused"
        assert need_head_weights, "attention maps are needed"
        with open(os.environ["FAKE_ESM_LOG"], "a") as log:
            log.write("forward\n")
        length = tokens.shape[1]
        generator = torch.Generator().manual_seed(length)
        return {"attentions": torch.rand(1, 33, 20, length, length, generator=generator)}


class FakeAlphabet:
    def get_batch_converter(self):
        def convert(data):
            (label, seq), = data
            for residue in seq:
                if residue not in ALPHABET:
                    raise KeyError(residue)
            return [label], [seq], torch.zeros(1, len(seq) + 2, dtype=torch.long)

        return convert


def esm1b_t33_650M_UR50S():
    return FakeEsm(), FakeAlphabet()
