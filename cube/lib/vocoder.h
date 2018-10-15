#include <vector>
#include "ml.h"

class Matrix;

class Vocoder{
    private:
        unsigned int sample_rate;
        unsigned int mgc_order;
        Matrix *upsample_w;
        Matrix *upsample_b;
        LSTM rnn_fine;
        LSTM rnn_coarse;
        Matrix mlp_coarse_w;
        Matrix mlp_coarse_b;
        Matrix mlp_fine_w;
        Matrix mlp_fine_b;
        Matrix hidden_coarse;//prealocated buffer
        Matrix hidden_fine;//prealocated buffer

        Matrix softmax_coarse_w;
        Matrix softmax_coarse_b;
        Matrix softmax_fine_w;
        Matrix softmax_fine_b;
        Matrix softmax_coarse;//prealocated buffer
        Matrix softmax_fine;//prealocated buffer
        Matrix upsample;//prealocated buffer


//        std::vector<Matrix> upsample(float *spec, int num_frames);
        int upsample_count;
        int sample(Matrix &softmax, float temp);
    public:
        Vocoder(unsigned int sample_rate, unsigned int mgc_order);
        ~Vocoder();
        int load_from_file(char *filename);
        int *vocode(double *spectrogram, int num_frames, float temperature);
};