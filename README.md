# PyTorch-Linear-Operator-CUDA

This is a simple demo of how to write a Linear operator (Y = XW+b) in PyTorch C++ CUDA extension. 

The corresponding blog (text explanation for the code) is at [here](https://i-am-future.github.io/2023/07/30/Pytorch-Practical-Basics-6/). (From section 6 to section 9 are all talking about that.)

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

    + (Unfortunately, our implementation are not as fast as PyTorch's, which utilizes cuBLAS and cuDNN. But it is still a good demo in how to write PyTorch C++ CUDA extension.)

+ A Python-side module interface (`torch.autograd.Function`) for matrix multiplication operator and a simple linear (Y = XW+b) operator.

+ Testings (manually `torch.allclose` and `torch.autograd.gradcheck`) to make sure the forward and backward are correct.

+ An short example of using our own linear layer vs. torch's linear layer in classifying MNIST dataset. Supprisingly, our implementation is even a little faster than torch's.

## Build

This project requires CUDA compiler installed. Use `nvcc -V` to check you have CUDA compiler installed.

We recommend you to build a new virtual environment for this project. 

```bash
conda create -n env_name
# install pytorch
const activate env_name
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then, install the project.


```bash
# change to project dir first.
python setup.py install
```

**Special Note:** If you want to run the python codes (in `examples` and `test`), you may need to adjust the `os.environ['CUDA_VISIBLE_DEVICES']` to an appropriate value, e.g., `'0'`. Sorry for the inconvenience. 


## Python interface usage

```python
import torch
import mylinearops

dir(mylinearops)
[..., 'linearop', 'matmul', 'mylinearops']
```


Or, you can try the example of classifying Mnist digits:

```bash
# use our implementation
python examples/main.py > examples/log

# use torch's implementation
python examples/main.py --torch > examples/log_torch
```

