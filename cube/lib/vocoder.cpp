#include <stdio.h>
#include "vocoder.h"
#include "ml.h"


Vocoder::Vocoder(unsigned int sample_rate, unsigned int mgc_order){
    int count=0;
    char **tmp=0x0;
    //dynet::initialize(count, tmp);
    printf("SAMPLE_RATE=%d\nMGC_ORDER=%d\n", sample_rate, mgc_order);
    this->sample_rate=sample_rate;
    this->mgc_order=mgc_order;

    const unsigned int UPSAMPLE_PROJ = 200;
    const unsigned int RNN_SIZE = 448;
    const unsigned int RNN_LAYERS = 1;


    this->upsample_count = int(12.5 * this->sample_rate / 1000);

    for (int i=0;i<upsample_count;i++){
        this->upsample_w.push_back(Matrix(UPSAMPLE_PROJ, this->mgc_order * 2));//(this->model.add_parameters({UPSAMPLE_PROJ, this->mgc_order * 2}));
        this->upsample_b.push_back(Matrix(UPSAMPLE_PROJ));//(this->model.add_parameters({UPSAMPLE_PROJ}));
    }
//    this->model.add_lookup_parameters(256,{1});//    this->model.add_lookup_parameters(256,{1});
//
//    this->rnn_coarse=dynet::VanillaLSTMBuilder(RNN_LAYERS, 2 + UPSAMPLE_PROJ, RNN_SIZE, this->model);
//    this->rnn_fine=dynet::VanillaLSTMBuilder(RNN_LAYERS, 3 + UPSAMPLE_PROJ, RNN_SIZE, this->model);
//
    this->mlp_coarse_w=Matrix(RNN_SIZE, RNN_SIZE);//this->model.add_parameters({RNN_SIZE, RNN_SIZE});
    this->mlp_coarse_b=Matrix(RNN_SIZE);//this->model.add_parameters({RNN_SIZE});
    this->mlp_fine_w=Matrix(RNN_SIZE, RNN_SIZE);//this->model.add_parameters({RNN_SIZE, RNN_SIZE});
    this->mlp_fine_b=Matrix(RNN_SIZE);//this->model.add_parameters({RNN_SIZE});
//
    this->softmax_coarse_w=Matrix(256, RNN_SIZE);//this->model.add_parameters({256, RNN_SIZE});
    this->softmax_coarse_b=Matrix(256);//this->model.add_parameters({256});
    this->softmax_fine_w=Matrix(256, RNN_SIZE);//this->model.add_parameters({256, RNN_SIZE});
    this->softmax_fine_b=Matrix(256);//this->model.add_parameters({256});
}

Vocoder::~Vocoder(){
}

int Vocoder::load_from_file(char *fn){
    printf("Loading %s\n", fn);
    std::ifstream f(fn);
    for (int i=0;i<this->upsample_w.size();i++){
        this->upsample_w[i].load_from_file(f);
        this->upsample_b[i].load_from_file(f);
    }
    //dynet::TextFileLoader l(fn);
    //l.populate(this->model);
    Matrix ll(256,1);//older model compatibility
    ll.load_from_file(f);
    ll.load_from_file(f);
    f.close();
    return 0;
}

std::vector <Matrix> Vocoder::upsample(float *spec, int num_frames){
    std::vector<Matrix> rez;
    std::vector<Matrix> input_data;
    for (int i=0;i<num_frames-1;i++){
//        int index1=i*this->mgc_order;
//        std::vector<dynet::real> x_values(this->mgc_order*2);
//        for (int cp=0;cp<this->mgc_order*2;cp++){
//            x_values[cp]=spec[index1+cp];
//        }
//        dynet::Expression x=input(cg, {this->mgc_order*2}, x_values);
//        input_data.push_back(x);
    }

    for (int i=0;i<num_frames-1;i++){
//        //printf("%f %f\n", dynet::as_vector(input_data[i].value())[0], dynet::as_vector(input_data[i].value())[this->mgc_order]);
//        for (int ups_index=0;ups_index<this->upsample_count;ups_index++){
//            rez.push_back(dynet::tanh(parameter(cg,this->upsample_w[ups_index])*input_data[i]+parameter(cg,this->upsample_b[ups_index])));
//        }
    }

    return rez;
}

int Vocoder::sample(Matrix softmax, float temp){
    //std::vector<float> values=dynet::as_vector(softmax.value());
    //printf (".");
    //fflush(stdout);
    return 0;
}

std::vector<int> Vocoder::synthesize(std::vector<Matrix> &upsampled, float temp){
    std::vector<int> audio;
    int total_audio_points=(int)(upsampled.size());
    printf("\tEstimated raw audio points is %d\n", total_audio_points);
    printf("\tGenerating raw audio: ");
    fflush(stdout);
    int last_proc=0;
//    this->rnn_coarse.new_graph(cg);
//    this->rnn_fine.new_graph(cg);
//    this->rnn_coarse.start_new_sequence();
//    this->rnn_fine.start_new_sequence();
//    float last_coarse_sample=0;
//    float last_fine_sample=0;
//    for (int i=0;i<total_audio_points;i++){
//        int curr_proc=(int)((i+1)*100/total_audio_points);
//        if (curr_proc % 5==0 && curr_proc!=last_proc){
//            printf("%d ", curr_proc);
//            last_proc=curr_proc;
//            fflush(stdout);
//        }
//        std::vector<dynet::real> coarse_input(2);
//        coarse_input[0]=last_coarse_sample/128.0-1.0;
//        coarse_input[1]=last_fine_sample/128.0-1.0;
//        dynet::Expression coarse_input_expr=input(cg, {2}, coarse_input);
//        std::vector<dynet::Expression> concat_list;
//        concat_list.push_back(coarse_input_expr);
//        concat_list.push_back(upsampled[i]);
//        dynet::Expression coarse_input_final=dynet::concatenate(concat_list);
//        dynet::Expression rnn_coarse_output=this->rnn_coarse.add_input(coarse_input_final);
//        dynet::Expression hidden_coarse=dynet::rectify(dynet::parameter(cg,this->mlp_coarse_w)*rnn_coarse_output+ dynet::parameter(cg,this->mlp_coarse_b));
//
//        dynet::Expression softmax_coarse=dynet::softmax(dynet::parameter(cg,this->softmax_coarse_w)*hidden_coarse+ dynet::parameter(cg,this->softmax_coarse_b));
//        int coarse_output=this->sample(softmax_coarse, temp);
//    }
//    printf("done\n");
    return audio;
}

int *Vocoder::vocode(double *spectrogram, double *mean, double *stdev, int num_frames, float temperature){
//    dynet::ComputationGraph cg;
//    float audio_length_ms=(12.5*num_frames)/1000;
//    printf("\tEstimated audio size is %f seconds\n", audio_length_ms);
//    float *f_spec=new float[this->mgc_order*num_frames];
//    int total=this->mgc_order*num_frames;
//    for (int i=0;i<total;i++){
//        f_spec[i]=(float)spectrogram[i];
//    }
//    std::vector<dynet::Expression> upsampled_spec=this->upsample(f_spec, num_frames, cg);
//    std::vector<int> audio=this->synthesize(upsampled_spec, temperature, cg);
//    delete []f_spec;
}