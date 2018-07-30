from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ['CFLAGS'] = '-O3 -Wall'

ext_runtime = Extension('cube_runtime', sources=["lib/cube_runtime.pyx", "lib/entry_point.cpp", "lib/vocoder.cpp", "lib/ml.cpp"],
                        language="c++",
                        extra_compile_args=["-std=c++0x", "-O3"],
                        include_dirs=[numpy.get_include()])

setup(name="cube_runtime", ext_modules=cythonize([ext_runtime]))
