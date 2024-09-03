from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_graph_extension',
    ext_modules=[
        CUDAExtension('cuda_graph_extension', [
            'cuda_graph_extension.cu', '/home/deepl/sooho/DKH-/cu2py_test/include/func.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=[
        '/home/deepl/sooho/DKH-/cu2py_test/include'
    ]
)
