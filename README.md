# ARC-Net: Enhancing Protein Function Prediction through an Atomic Environment and Residue Graph Co-learning Networ

This is the official PyTorch implementation of our paper, **"ARC-Net: Enhancing Protein Function Prediction through an Atomic Environment and Residue Graph Co-learning Network"**.

---

## Installation

This codebase is based on PyTorch and [TorchDrug] ([TorchProtein](https://torchprotein.ai)). 
It supports training and inference with multiple GPUs.
The documentation and implementation of our methods can be found in the [docs](https://torchdrug.ai/docs/) of TorchDrug.
To adapt our model in your setting, you can follow the step-by-step [tutorials](https://torchprotein.ai/tutorials) in TorchProtein.

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


## Training and evaluating

We provide scripts for training and evaluating the model on EC and GO tasks

**Train the model for the EC task:**
```bash
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_ec.py'
```

**Train the model for the GO task:**
```bash
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_go.py --go BP'
sh -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_go.py --go MF'
```



## Acknowledgements

* The code implementation and environment setup of this project are partially based on [GearNet](https://github.com/DeepGraphLearning/GearNet). We sincerely thank the original authors for their open-source contributions.
