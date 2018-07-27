#include <stdio.h>
#include <Python.h>
#include <string>
#include "vocoder.h"

Vocoder *vocoder;

void c_print_version(){
    printf("TTS-Cube Runtime version 0.9beta\n");
}


int c_load_vocoder(char* path){
    char *fn=new char[1024];
    snprintf(fn,1024, "%s.network", path);
    vocoder = new Vocoder((unsigned int)16000, (unsigned int)60);
    vocoder->   load_from_file(fn);
    delete []fn;
    return 0;
}
