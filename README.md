# ARC-Net: Enhancing Protein Function Prediction through an Atomic Environment and Residue Graph Co-learning Networ

This is the official PyTorch implementation of our paper, **"ARC-Net: Enhancing Protein Function Prediction through an Atomic Environment and Residue Graph Co-learning Network"**.

---

## Installation

This codebase is built upon PyTorch and [TorchDrug] ([TorchProtein](https://torchprotein.ai)). It natively supports both training and inference on multiple GPUs.
You can install the required dependencies using either Conda or Pip. This code has been tested with Python 3.7/3.8 and PyTorch 1.8.0 (LTS).

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

## Installation

You may install the dependencies via either conda or pip. Generally, GearNet works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda

```bash
conda install torchdrug pytorch=1.8.0 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install easydict pyyaml -c conda-forge
```

### From Pip

```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install easydict pyyaml
```

## Dataset 
The Enzyme Commission (EC) and Gene Ontology (GO) datasets are sourced directly from the TorchDrug platform.
They will be automatically downloaded to your data directory the first time you run a training script. No manual data preparation is required.


## Training and evaluating

We provide scripts for training and evaluating the model on EC and GO tasks
The following scripts demonstrate how to train and evaluate the ARC-Net model on multiple GPUs using ```torch.distributed```

###EC Number Prediction
```bash
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_ec.py'
```
###Gene Ontology (GO) Prediction
**Train the model for the GO task:**
```bash
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_go.py --go BP'
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_go.py --go MF'
```



## Acknowledgements

* The code implementation and environment setup of this project are partially based on [GearNet](https://github.com/DeepGraphLearning/GearNet). We sincerely thank the original authors for their open-source contributions.
* This work is powered by the excellent [TorchDrug] ([TorchProtein](https://torchprotein.ai)) library.
