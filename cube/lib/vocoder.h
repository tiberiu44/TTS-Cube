#include <dynet/io.h>
#include <dynet/lstm.h>
#include <vector>

class Vocoder{
    private:
        dynet::ParameterCollection model;
        unsigned int sample_rate;
        unsigned int mgc_order;
        std::vector<dynet::Parameter> upsample_w;
        std::vector<dynet::Parameter> upsample_b;
        dynet::VanillaLSTMBuilder rnn_fine;
        dynet::VanillaLSTMBuilder rnn_coarse;
        dynet::Parameter mlp_coarse_w;
        dynet::Parameter mlp_coarse_b;
        dynet::Parameter mlp_fine_w;
        dynet::Parameter mlp_fine_b;

        dynet::Parameter softmax_coarse_w;
        dynet::Parameter softmax_coarse_b;
        dynet::Parameter softmax_fine_w;
        dynet::Parameter softmax_fine_b;
    public:
        Vocoder(unsigned int sample_rate, unsigned int mgc_order);
        ~Vocoder();
        int load_from_file(char *filename);
        int *vocode(double *spectrogram, double *mean, double *stdev, int num_frames, float temperature);

};