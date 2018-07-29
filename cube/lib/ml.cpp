#include <stdio.h>
#include <fstream>
#include <string.h>
#include <sstream>
#include <string>

#include "ml.h"

Matrix::Matrix(int rows, int cols){
    this->data=new double[rows*cols];
    this->rows=rows;
    this->cols=cols;
}

Matrix::Matrix(int rows){
    this->data=new double[rows];
    this->rows=rows;
    this->cols=1;
}

Matrix::~Matrix(){
    if (this->data!=0x0){
        delete []data;
        this->data=0x0;
    }
}

Matrix::Matrix(){
    this->data=0x0;
    this->rows=0;
    this->cols=0;
}

Matrix::Matrix(const Matrix &other){
        this->data=new double[other.cols*other.rows];
        this->cols=other.cols;
        this->rows=other.rows;
        memcpy(this->data, other.data, cols*rows*sizeof(double));
}



void Matrix::load_from_file (std::ifstream &f){
    std::string line;
    std::getline(f, line);
    std::getline(f, line);
    std::istringstream in(line.c_str());
    int total=cols*rows;
    for (int i=0;i<total;i++){
        in>>data[i];
    }
}