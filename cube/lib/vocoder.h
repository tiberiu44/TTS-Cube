#include <vector>
#include "ml.h"


class Vocoder{
    private:
        unsigned int sample_rate;
        unsigned int mgc_order;
        std::vector<Matrix> upsample_w;
        std::vector<Matrix> upsample_b;
        //dynet::VanillaLSTMBuilder rnn_fine;
        //dynet::VanillaLSTMBuilder rnn_coarse;
        Matrix mlp_coarse_w;
        Matrix mlp_coarse_b;
        Matrix mlp_fine_w;
        Matrix mlp_fine_b;

        Matrix softmax_coarse_w;
        Matrix softmax_coarse_b;
        Matrix softmax_fine_w;
        Matrix softmax_fine_b;

        std::vector<Matrix> upsample(float *spec, int num_frames);
        std::vector<int> synthesize(std::vector<Matrix> &upsampled, float temp);
        int upsample_count;
        int sample(Matrix softmax, float temp);
    public:
        Vocoder(unsigned int sample_rate, unsigned int mgc_order);
        ~Vocoder();
        int load_from_file(char *filename);
        int *vocode(double *spectrogram, double *mean, double *stdev, int num_frames, float temperature);

};