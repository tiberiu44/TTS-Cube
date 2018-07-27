#include <stdio.h>
#include <string>

extern void c_print_version();

extern int c_load_vocoder(char* path);

extern int* c_vocode(double *spectrogram, double *mean, double *stdev, int num_frames, float temperature);
