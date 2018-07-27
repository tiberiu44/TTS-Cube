#include <dynet/io.h>

dynet::ParameterCollection vocoder_collection;

dynet::ParameterCollection load_vocoder_from_file(char *filename){
    const int UPSAMPLE_PROJ = 200
    const int RNN_SIZE = 448
    const int RNN_LAYERS = 1
    const int OUTPUT_EMB_SIZE = 1

    const int target_sample_rate=16000 //should be passed as a parameter

    int upsample_count = int(12.5 * target_sample_rate / 1000)
    return vocoder_collection;
}