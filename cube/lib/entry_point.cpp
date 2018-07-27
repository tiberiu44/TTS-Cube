#include <stdio.h>
#include <Python.h>
#include <string>
#include "vocoder.h"

dynet::ParameterCollection vocoder;

void c_print_version(){
    printf("TTS-Cube Runtime version 0.9beta\n");
}


int c_load_vocoder(char* path){
    printf("\tLoading vocoder from '%s'\n", path);
    char *fn=new char[1024];
    snprintf(fn,1024, "%s.network", path);
    printf("\t\t setting path to %s\n", fn);
    vocoder = load_vocoder_from_file(fn);
    delete []fn;
    return 0;
}
