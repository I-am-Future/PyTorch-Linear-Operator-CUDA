import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

SRC_DIR = osp.join(ROOT_DIR, "src")
sources = glob.glob(osp.join(SRC_DIR, '*.cpp'))+glob.glob(osp.join(SRC_DIR, '*.cu'))


setup(
    name='mylinearops',
    version='1.0',
    author='I-am-Future',
    author_email='3038242641@qq.com',
    description='Hand-written Linear ops for PyTorch',
    long_description='Simple demo for writing Linear ops in CUDA extensions with PyTorch',
    ext_modules=[
        CUDAExtension(
            name='mylinearops_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    py_modules=['mylinearops.mylinearops'],
    cmdclass={
        'build_ext': BuildExtension
    }
)