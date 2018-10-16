#include <stdio.h>
#include <time.h>
#include "vocoder.h"
#include "ml.h"
#include <math.h>


Vocoder::Vocoder(unsigned int sample_rate, unsigned int mgc_order){
    printf("SAMPLE_RATE=%d\nMGC_ORDER=%d\n", sample_rate, mgc_order);
    this->sample_rate=sample_rate;
    this->mgc_order=mgc_order;

    const unsigned int UPSAMPLE_PROJ = 200;
    const unsigned int RNN_SIZE = 448;
    //const unsigned int RNN_LAYERS = 1;


    this->upsample_count = int(12.5 * this->sample_rate / 1000);

    this->upsample_w=new Matrix[upsample_count];
    this->upsample_b=new Matrix[upsample_count];
    for (int i=0;i<upsample_count;i++){
        this->upsample_w[i]=Matrix(UPSAMPLE_PROJ, this->mgc_order*2);
        this->upsample_b[i]=Matrix(UPSAMPLE_PROJ);
    }

    this->rnn_coarse=LSTM(2 + UPSAMPLE_PROJ, RNN_SIZE);
    this->rnn_fine=LSTM(3 + UPSAMPLE_PROJ, RNN_SIZE);

    this->mlp_coarse_w=Matrix(RNN_SIZE, RNN_SIZE);
    this->mlp_coarse_b=Matrix(RNN_SIZE);
    this->mlp_fine_w=Matrix(RNN_SIZE, RNN_SIZE);
    this->mlp_fine_b=Matrix(RNN_SIZE);

    this->softmax_coarse_w=Matrix(256, RNN_SIZE);
    this->softmax_coarse_b=Matrix(256);
    this->softmax_fine_w=Matrix(256, RNN_SIZE);
    this->softmax_fine_b=Matrix(256);

    //prealocated matrices
    this->hidden_coarse=Matrix(RNN_SIZE);
    this->hidden_fine=Matrix(RNN_SIZE);
    this->softmax_fine=Matrix(256);
    this->softmax_coarse=Matrix(256);
    this->upsample=Matrix(UPSAMPLE_PROJ+3);//+3 is a trick to avoid copying memory during synthesis

}

Vocoder::~Vocoder(){
}

int Vocoder::load_from_file(char *fn){
    printf("Loading %s\n", fn);
    std::ifstream f(fn);
    for (int i=0;i<this->upsample_count;i++){
        this->upsample_w[i].load_from_file(f);
        this->upsample_b[i].load_from_file(f);
    }
    //dynet::TextFileLoader l(fn);
    //l.populate(this->model);
    Matrix ll(1,256);//older model compatibility


    this->rnn_coarse.load_from_file(f);
    this->rnn_fine.load_from_file(f);
    this->mlp_coarse_w.load_from_file(f);
    this->mlp_coarse_b.load_from_file(f);
    this->mlp_fine_w.load_from_file(f);
    this->mlp_fine_b.load_from_file(f);
    this->softmax_coarse_w.load_from_file(f);
    this->softmax_coarse_b.load_from_file(f);
    this->softmax_fine_w.load_from_file(f);
    this->softmax_fine_b.load_from_file(f);

    ll.load_from_file(f);
    ll.load_from_file(f);
    f.close();
    return 0;
}

int Vocoder::sample(Matrix &layer, float temp){
//    double sum=0;
//    double max=layer.data[0];
//    for (int i=1;i<layer.rows;i++){
//        if (layer.data[i]>max){
//            max=layer.data[i];
//        }
//    }
//    for (int i=0;i<layer.rows;i++){
//        layer.data[i]=exp(layer.data[i]-max);
//        sum+=layer.data[i];
//    }
    int max_index=0;
    for (int i=0;i<layer.rows;i++){
        //layer.data[i]/=sum;
        if (layer.data[i]>layer.data[max_index])
            max_index=i;
    }
    //layer.print();


    return max_index;
}

int *Vocoder::vocode(double *spectrogram, int num_frames, float temperature){
    //exit(0);
    clock_t start,end;
    start=clock();

    float audio_length_ms=(12.5*num_frames)/1000;
    printf("\tEstimated audio size is %f seconds\n", audio_length_ms);
    std::vector<int> audio;
    int total_audio_points=(int)(audio_length_ms*this->sample_rate);
    printf("\tEstimated raw audio points is %d\n", total_audio_points);
    printf("\tGenerating raw audio: ");
    fflush(stdout);
    int last_proc=0;
    int index=0;
    int last_coarse_sample=0;
    int last_fine_sample=0;

    Matrix input_cond=Matrix(this->mgc_order*2);
    int cnt=0;

    this->rnn_fine.reset();
    this->rnn_coarse.reset();

    //printf ("NUM FRAMES=%d\nupsample_count=%d\n", num_frames, upsample_count);
    for (int i=0;i<num_frames-1;i++){
        //printf("i=%d\n",i);
        //create conditioning input matrix
        memcpy(input_cond.data, spectrogram+i*this->mgc_order, sizeof(double)*this->mgc_order*2);
        for (int j=0;j<this->upsample_count;j++){
            //printf("\t\tj=%d\n",j);
            index++;
            int curr_proc=index*100/total_audio_points;
            if (curr_proc%5==0 && curr_proc!=last_proc){
                printf("%d ", curr_proc);
                fflush(stdout);
                last_proc=curr_proc;
            }
            double *orig_ptr=upsample.data;
            int num_rows=upsample.rows;
            upsample.data=&orig_ptr[3];//move the pointer to prepare for sampling
            upsample.rows=num_rows-3;
            upsample_w[j].multiply(input_cond, upsample);
            upsample.add(upsample_b[j], upsample);
            upsample.apply_tanh();
            //upsample.print();
            //exit(0);
            upsample.data=&orig_ptr[1];//move back the pointer for coarse synthesis
            upsample.rows=num_rows-1;
            upsample.data[0]=(float)last_coarse_sample/128.0-1.0;
            upsample.data[1]=(float)last_fine_sample/128.0-1.0;

            rnn_coarse.add_input(upsample);
            mlp_coarse_w.multiply(rnn_coarse.ht, hidden_coarse);
            hidden_coarse.add(mlp_coarse_b, hidden_coarse);
            hidden_coarse.apply_rectify();

            softmax_coarse_w.multiply(hidden_coarse, softmax_coarse);
            softmax_coarse.add(softmax_coarse_b, softmax_coarse);

            upsample.data=orig_ptr;//move back the pointer for fine synthesis
            upsample.rows=num_rows;
            //printf(".\n");
            //printf ("coarse\n");
            int selected_coarse_sample=this->sample(softmax_coarse, temperature);
            upsample.data[0]=(float)last_coarse_sample/128.0-1.0;
            upsample.data[1]=(float)last_fine_sample/128.0-1.0;
            upsample.data[2]=(float)selected_coarse_sample/128.0-1.0;

            rnn_fine.add_input(upsample);
            mlp_fine_w.multiply(rnn_fine.ht, hidden_fine);
            hidden_fine.add(mlp_fine_b, hidden_fine);
            hidden_fine.apply_rectify();
            softmax_fine_w.multiply(hidden_fine, softmax_fine);
            softmax_fine.add(softmax_fine_b, softmax_fine);
            //printf ("fine\n");
            int selected_fine_sample=this->sample(softmax_fine, temperature);

            last_coarse_sample=selected_coarse_sample;
            last_fine_sample=selected_fine_sample;


            audio.push_back(((long)last_coarse_sample * 256 + last_fine_sample)-32768);
            //printf("%d ", audio[audio.size()-1]);
        }
    }
    end=clock();
    double dif = difftime (end,start)*1.0/CLOCKS_PER_SEC;
    printf("done in %f seconds with %d\n", dif, cnt);
    int *a =new int[audio.size()];
    for (unsigned int i=0;i<audio.size();i++){
        a[i]=audio[i];
        //a[i]=i;
    }
    return a;

}