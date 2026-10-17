import os
import sys
from setuptools import setup, Extension
import pybind11

# Audit B8: build from the repository root so that "src/exchange.cpp" is a relative path.
# setuptools cannot write object files for a source path that starts with "..".
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

ext_modules = [
    Extension(
        "cpp_exchange",                  
        # Audit B8: the source is in src/, not in python/.
        ["src/exchange.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        # THE FIX: Aggressive MSVC hardware optimizations
        extra_compile_args=['/O2', '/fp:fast', '/arch:AVX2', '/GL'] if sys.platform == 'win32' else ['-O3', '-march=native', '-ffast-math']
    ),
]

setup(
    name="cpp_exchange",
    ext_modules=ext_modules,
    # With --inplace, the compiled module goes into python/, next to the scripts that import it.
    package_dir={"": "python"},
)