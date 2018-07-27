# distutils: language = c++
cimport cython

cdef extern from "entry_point.h":
    cdef void c_print_version()

def print_version():
    c_print_version()

