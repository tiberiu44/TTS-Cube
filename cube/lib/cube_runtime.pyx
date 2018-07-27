# distutils: language = c++
cimport cython
from ctypes import *


cdef extern from "entry_point.h":
    cdef void c_print_version()

cdef extern from "entry_point.h":
    cdef int c_load_vocoder(char *path)

def load_vocoder(path):
    import array
    path=path.encode('utf-8')
    rez= c_load_vocoder(<bytes>path)
    return rez


def print_version():
    c_print_version()

