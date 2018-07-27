from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_runtime = Extension('cube_runtime', sources=["lib/cube_runtime.pyx", "lib/entry_point.cpp"], language="c++")

setup(name="cube_runtime", ext_modules = cythonize([ext_runtime]))
