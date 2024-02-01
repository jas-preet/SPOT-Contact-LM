SPOT-Contact-Single: *Improving Single-Sequence-Based Prediction of Protein Contact Map using a Transformer Language Model.*
====
The standalone version / CLI-tool of SPOT-Contact-Single available for public use for research purposes.

Contents
----
  * [Introduction](#introduction)
  * [Results](#results)
  * [System Requirments](#system-requirments)

# SPOT-Contact-Single

Introduction
----
Accurate prediction of protein contact-map is essential for accurate protein structure and function prediction. As a result, many methods have been developed for protein contact map prediction. However, most methods rely on protein-sequence-evolutionary information, which may not exist for many proteins due to lack of naturally occurring homologous sequences. Moreover, generating evolutionary profiles is computationally intensive. Here, we developed a contact-map predictor utilizing the output of a pre-trained language model ESM-1B as an input along with a large training set and an ensemble of residual neural networks.


Results
----
We showed that the proposed method makes a significant improvement over a single-sequence-based predictor SSCpred with 15% improvement in the F1-score for the independent CASP14-FM test set. It also outperforms evolutionary-profile-based methods TrRosetta and SPOT-Contact with 48.7% and 48.5% respective improvement in the F1-score on the proteins without homologs (Neff=1) in the independent SPOT-2018 set. The new method provides a much faster and reasonably accurate alternative to evolution-based methods, useful for large-scale prediction.

System Requirments
----

**Hardware Requirments:**
SPOT-Contact-single predictor has been tested on standard ubuntu 18 computer with approximately 32 GB RAM to support the in-memory operations.

* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive) (Optional if using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional if using GPU)

Installation
----

To install SPOT-Contact-Single and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/jas-preet/SPOT-Contact-Single.git`
2. `cd SPOT-Contact-Single`

To download the model check points from the dropbox use the following commands in the terminal:

3. `wget https://apisz.sparks-lab.org:8443/downloads/Resource/Protein/2_Protein_local_structure_prediction/contact_jits.tar.xz`
4. `tar -xvf contact_jits.tar.xz`

To install the dependencies and create a conda environment use the following commands

5. `conda create -n spot_contact_sgl python=3.7`
6. `conda activate spot_contact_sgl`

if GPU computer:
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`

for CPU only 
7. `conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch`

8. `pip install fair-esm`

9. `conda install pandas=1.1.1`

10. `conda install tqdm`

Input dependency
----
SPOT-Contact-Single uses the output of **SPOT-1D-Single**. Please generate the prediction of SPOT-1D-Single from the Stand-alone version available at `https://github.com/jas-preet/SPOT-1D-Single`.
Copy the prediction for the required proteins to `data_files/out_spot_1d_sgl/`

Execute
----
To run SPOT-Contact-Single use the following command

`python -W ignore spot_contact_single.py --file_list file_lists/file_list_casp14.txt --save_path results/ --device cuda:0 --esm_device cuda:0`

or 

`python -W ignore spot_contact_single.py --file_list file_lists/file_list_casp14.txt --save_path results/ --device cpu --esm_device cpu` 

License
----

MIT License

for more details on this work refer the manuscript

Copyright (c) 2011-2017 GitHub Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
