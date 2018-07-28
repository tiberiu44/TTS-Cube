#include <stdio.h>
#include <dynet/io.h>
#include <dynet/lstm.h>
#include "vocoder.h"


Vocoder::Vocoder(unsigned int sample_rate, unsigned int mgc_order){
    printf("SAMPLE_RATE=%d\nMGC_ORDER=%d\n", sample_rate, mgc_order);
    this->sample_rate=sample_rate;
    this->mgc_order=mgc_order;

    const unsigned int UPSAMPLE_PROJ = 200;
    const unsigned int RNN_SIZE = 448;
    const unsigned int RNN_LAYERS = 1;


    int upsample_count = int(12.5 * this->sample_rate / 1000);

    for (int i=0;i<upsample_count;i++){
        this->upsample_w.push_back(this->model.add_parameters({UPSAMPLE_PROJ, this->mgc_order * 2}));
        this->upsample_b.push_back(this->model.add_parameters({UPSAMPLE_PROJ}));
    }

    this->model.add_lookup_parameters(256,{1});
    this->model.add_lookup_parameters(256,{1});

    this->rnn_coarse=dynet::VanillaLSTMBuilder(RNN_LAYERS, 2 + UPSAMPLE_PROJ, RNN_SIZE, this->model);
    this->rnn_fine=dynet::VanillaLSTMBuilder(RNN_LAYERS, 3 + UPSAMPLE_PROJ, RNN_SIZE, this->model);

    this->mlp_coarse_w=this->model.add_parameters({RNN_SIZE, RNN_SIZE});
    this->mlp_coarse_b=this->model.add_parameters({RNN_SIZE});
    this->mlp_fine_w=this->model.add_parameters({RNN_SIZE, RNN_SIZE});
    this->mlp_fine_b=this->model.add_parameters({RNN_SIZE});

    this->softmax_coarse_w=this->model.add_parameters({256, RNN_SIZE});
    this->softmax_coarse_b=this->model.add_parameters({256});
    this->softmax_fine_w=this->model.add_parameters({256, RNN_SIZE});
    this->softmax_fine_b=this->model.add_parameters({256});
}

int Vocoder::load_from_file(char *fn){
    printf("Loading %s\n", fn);
    dynet::TextFileLoader l(fn);
    l.populate(this->model);
    return 0;
}
int *Vocoder::vocode(double *spectrogram, double *mean, double *stdev, int num_frames, float temperature){
}