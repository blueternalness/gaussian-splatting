from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dequantize_cuda',
    ext_modules=[
        CUDAExtension('dequantize_cuda', [
            'dequantize.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })