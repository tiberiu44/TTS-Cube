# distutils: language = c++
cimport cython
from ctypes import *


cdef extern from "entry_point.h":
    cdef void c_print_version()

cdef extern from "entry_point.h":
    cdef int c_load_vocoder(char *path)

cdef extern from "entry_point.h":
    cdef int *c_vocode(double *spectogram, double *mean, double *stdev, int num_frames, float temperature)

def load_vocoder(path):
    import array
    path=path.encode('utf-8')
    rez= c_load_vocoder(<bytes>path)
    return rez

def vocode(spectogram, mean, stdev, temperature=1.0):
    c_vocode(<double*> input.data, <double*> mean.data, <double*> stdev.data, spectogram.shape[0], temperature)



def print_version():
    c_print_version()

