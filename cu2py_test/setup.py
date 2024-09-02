from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ipc_extension',
    ext_modules=[
        CUDAExtension('ipc_extension', [
            'cu2py_ptr_read.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
