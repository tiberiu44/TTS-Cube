from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ['CFLAGS'] = '-O3 -Wall'

ext_runtime = Extension('cube_runtime', sources=["lib/cube_runtime.pyx", "lib/entry_point.cpp", "lib/vocoder.cpp", "lib/ml.cpp"],
                        language="c++",
                        #libraries=['mkl_rt'],
                        extra_compile_args=["-std=c++0x", "-O3", "-march=native"],
                        extra_link_args=['/opt/intel/mkl/lib/intel64/libmkl_rt.so'],
                        include_dirs=[numpy.get_include(), '/opt/intel/mkl/include/'])

setup(name="cube_runtime", ext_modules=cythonize([ext_runtime]))
