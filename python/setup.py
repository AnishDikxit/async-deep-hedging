import os
import sys
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cpp_exchange",                  
        ["exchange.cpp"],                
        include_dirs=[pybind11.get_include()],
        language='c++',
        # THE FIX: Aggressive MSVC hardware optimizations
        extra_compile_args=['/O2', '/fp:fast', '/arch:AVX2', '/GL'] if sys.platform == 'win32' else ['-O3', '-march=native', '-ffast-math']
    ),
]

setup(
    name="cpp_exchange",
    ext_modules=ext_modules,
)