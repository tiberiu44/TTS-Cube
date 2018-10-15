# distutils: language = c++
cimport cython
from ctypes import *
import numpy as np
cimport numpy as np


cdef extern from "entry_point.h":
    cdef void c_print_version()

cdef extern from "entry_point.h":
    cdef int c_load_vocoder(char *path)

cdef extern from "entry_point.h":
    cdef int* c_vocode(double* spectrogram,  int num_frames, int frame_size, float temperature)

def load_vocoder(path):
    import array
    path=path.encode('utf-8')
    rez= c_load_vocoder(<bytes>path)
    return rez

def vocode(np.ndarray[double, ndim=2, mode="c"] spec, temperature=1.0):
    num_frames=spec.shape[0]
    frame_size=spec.shape[1]
    dim=(int)((num_frames-1)*12.5*16)
    #dim=int(12.5*16)
    cdef int[::1] view = <int[:dim]> c_vocode(&spec[0,0], num_frames, frame_size, temperature)
    return np.asarray(view)

def print_version():
    c_print_version()

