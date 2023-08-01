# PyTorch-Linear-Operator-CUDA

This is a simple demo of how to write a Linear operator (Y = XW+b) in PyTorch C++ CUDA extension. 

It includes:

+ An entire template of how to write a CUDA extension in PyTorch. This includes:

    + How to write `setup.py`
    
    + How to write `PYBIND11` code to bind C++ and Python

    + How to implement CUDA kernel functions

    + How to write Python module wrapper

+ A bunch of C++/CUDA functions calculate the forward and backward of a linear operator. This includes:

    + Matrix multiplication (2 versions)

    + Matrix transpose (2 versions)

    + Matrix vector addition

    + Matrix sum along axis

    + (Unfortunately, our implementation are not as fast as PyTorch's, which utilizes cuBLAS and cuDNN. But it is a good demo in writing PyTorch C++ CUDA extension.)

+ A Python-side module interface for matrix multiplication operator and a simple linear (Y = XW+b) operator.

+ Testings (manually and `torch.autograd.gradcheck`) to make sure the forward and backward are correct.

+ An short example of using our own linear layer vs. torch's linear layer in classifying MNIST dataset. Supprisingly, our implementation is even a little faster than torch's.

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

dir(mylinearops)
[..., 'linearop', 'matmul', 'mylinearops']
```

Or, you can run the example we provide:

```bash
# use our implementation
python examples/main.py > examples/log

# use torch's implementation
python examples/main.py --torch > examples/log_torch
```

