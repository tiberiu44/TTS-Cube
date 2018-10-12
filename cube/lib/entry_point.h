#include <stdio.h>
#include <string>
#include <numpy/ndarraytypes.h>

extern void c_print_version();

extern int c_load_vocoder(char* path);

extern int* c_vocode(double* spectrogram, int num_frames, int frame_size, float temperature);
