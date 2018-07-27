from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_runtime = Extension('cube_runtime', sources=["lib/cube_runtime.pyx", "lib/entry_point.cpp", "lib/vocoder.cpp"],
                        language="c++", include_dirs=['/usr/local/include/eigen3/'], extra_compile_args=["-std=c++0x"])

setup(name="cube_runtime", ext_modules=cythonize([ext_runtime]))
