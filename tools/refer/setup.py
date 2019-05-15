"""
This code is for making mask.so, used to visualize the segmentation of referred object.
All "mask" related code is copied from https://github.com/pdollar/coco.git
"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [
            Extension(
                'external._mask',
                sources=['external/maskApi.c', 'external/_mask.pyx'],
                include_dirs = [np.get_include(), 'external'],
                extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
            )
        ]

setup(
    name='external',
    packages=['external'],
    package_dir = {'external': 'external'},
    version='2.0',
    ext_modules=cythonize(ext_modules)
    )
