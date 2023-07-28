# PyTorch-Linear-Operator-CUDA

This is a simple demo of how to write a Linear operator (Y = XW+b) in PyTorch C++ CUDA extension. 

It includes:

+ An entire template of how to write a CUDA extension in PyTorch. This includes:

    + How to write `setup.py`
    
    + How to write `PYBIND11` code to bind C++ and Python

    + How to implement CUDA kernel functions

    + How to write Python module wrapper

+ A simple Python matrix multiplication operator and a simple linear (Y = XW+b) operator.

+ A bunch of C++/CUDA functions calculate the forward and backward of a linear operator. This includes:

    + Matrix multiplication (2 versions)

    + Matrix transpose (2 versions)

+ Testings to make sure the forward and backward are correct.

## Build

This project requires CUDA compiler installed. Use `nvcc -V` to check you have CUDA compiler installed.

We recommend you to build a new virtual environment for this project. 

```bash
conda create -n env_name
# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
const activate env_name
```

Then, install the project.


```bash
python setup.py install
```


## Python interface usage

```python
import torch
import mylinearops
```
