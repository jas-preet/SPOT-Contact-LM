# SPOT-Contact-Single

*Improving Single-Sequence-Based Prediction of Protein Contact Map using a Transformer Language Model.*

The standalone version of SPOT-Contact-Single, available for public use for research purposes.

## Contents

- [Introduction](#introduction)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Limitations](#limitations)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

## Introduction

Accurate prediction of protein contact-map is essential for accurate protein structure and function prediction. As a result, many methods have been developed for protein contact map prediction. However, most methods rely on protein-sequence-evolutionary information, which may not exist for many proteins due to lack of naturally occurring homologous sequences. Moreover, generating evolutionary profiles is computationally intensive. Here, we developed a contact-map predictor utilizing the output of a pre-trained language model ESM-1B as an input along with a large training set and an ensemble of residual neural networks.

## Results

We showed that the proposed method makes a significant improvement over a single-sequence-based predictor SSCpred with 15% improvement in the F1-score for the independent CASP14-FM test set. It also outperforms evolutionary-profile-based methods TrRosetta and SPOT-Contact with 48.7% and 48.5% respective improvement in the F1-score on the proteins without homologs (Neff=1) in the independent SPOT-2018 set. The new method provides a much faster and reasonably accurate alternative to evolution-based methods, useful for large-scale prediction.

## Requirements

- Python 3 (tested with 3.12) and the packages in `requirements.txt`
- About 8 GB of disk space, because ESM-1b is downloaded the first time it runs
- Memory on CPU that grows with the square of the sequence length, as shown below
- Optional: an NVIDIA GPU with CUDA

| Sequence length | Estimated memory on CPU |
| --- | --- |
| 100 residues | 9 GB |
| 200 residues | 14 GB |
| 300 residues | 21 GB |
| 400 residues | 31 GB |
| 1,000 residues | 153 GB |

These figures come from a Mac using a PyTorch build without oneDNN, and other systems may use less. The largest measured run was 27.6 GB, for 404 residues. The program estimates this before it starts, and stops if the estimate is more than three quarters of your RAM. To override that, use `--max_memory_gb`, but only if you are sure the memory is available, because running out can make the computer unresponsive.

## Installation

```bash
git clone https://github.com/jas-preet/SPOT-Contact-LM.git
cd SPOT-Contact-LM
pip install -r requirements.txt
```

Download the model checkpoints (about 100 MB) and check the download. The server only offers plain HTTP, so don't skip the checksum.

```bash
curl -LO http://183.36.5.251:8080/sparks_downloads/servers/downloads/Resource/Protein/2_Protein_local_structure_prediction/contact_jits.tar.xz
echo "fb68555f977543f0f31e30d845f9c927cfb1216fd63dd263bc86a93feecf9e81  contact_jits.tar.xz" | shasum -a 256 -c
tar -xf contact_jits.tar.xz
```

On Linux, use `sha256sum -c` instead of `shasum -a 256 -c`. If the link stops working, look for this tool on the [Zhou lab download page](https://zhouyq-lab.szbl.ac.cn/download/).

## Usage

SPOT-Contact-Single uses the output of [SPOT-1D-Single](https://github.com/jas-preet/SPOT-1D-Single). Run SPOT-1D-Single on your proteins first, then copy its CSV files to `data_files/out_spot_1d_sgl/` or point `--spot_single_dir` at them. The repository includes this output for the CASP14 examples.

```bash
python -W ignore spot_contact_single.py --file_list file_list/test_file_list.txt --save_path results/ --device cpu --esm_device cpu
```

On a GPU, use `--device cuda:0 --esm_device cuda:0`. On CPU, `file_list/file_list_casp14.txt` includes a 404-residue protein, so it needs `--max_memory_gb 32` on computers with less than 42 GB of RAM.

| Option | Description |
| --- | --- |
| `--file_list` | Required. Text file with one FASTA path per line. Relative paths are resolved from the current directory. |
| `--save_path` | Output directory, created if needed. Default: `results/` |
| `--device` | `cpu` (default) or `cuda:0` |
| `--esm_device` | Device for ESM-1b, such as `cpu` (default) or `cuda:0` |
| `--jits_dir` | Directory holding the checkpoints. Default: `contact_jits/` next to the script |
| `--spot_single_dir` | Directory holding SPOT-1D-Single output. Default: `data_files/out_spot_1d_sgl/` next to the script |
| `--max_memory_gb` | Memory limit for the CPU estimate. Default: three quarters of RAM |

Each FASTA file must contain one sequence. Lower-case letters are accepted, a trailing `*` is removed, and letters ESM-1b cannot encode, such as `J`, are treated as unknown. Proteins are identified by FASTA file name, so names in a list must be unique.

## Output

Each input produces `<save_path>/<fasta file name>.npy`, an L x L NumPy array of contact probabilities, where L is the sequence length. Load it with `numpy.load`.

## Limitations

- ESM-1b accepts up to 1,022 residues, so longer sequences are rejected.
- Only single chains are supported, so multimers and sequences with separators such as `:` are rejected.
- The GPU checkpoints only run on `cuda:0`, because of how they were exported with `torch.jit.trace`.

## Tests

The tests use small stand-in models, so they don't need any downloads.

```bash
pip install pytest
pytest
```

## Citation

Singh J, Litfin T, Singh J, Paliwal K, Zhou Y. SPOT-Contact-LM: improving single-sequence-based prediction of protein contact map using a transformer language model. *Bioinformatics*. 2022;38(7):1888-1894. [doi:10.1093/bioinformatics/btac053](https://doi.org/10.1093/bioinformatics/btac053)

Please also cite [ESM-1b](https://github.com/facebookresearch/esm) and [SPOT-1D-Single](https://github.com/jas-preet/SPOT-1D-Single).

## License

MIT License

for more details on this work refer the manuscript

Copyright (c) 2011-2017 GitHub Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
