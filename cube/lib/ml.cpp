#include <stdio.h>
#include <fstream>
#include <string.h>
#include <sstream>
#include <memory.h>
#include <math.h>
#include <mkl.h>

#include "ml.h"

Matrix::Matrix(int rows, int cols){
    this->data=new double[rows*cols];
    this->rows=rows;
    this->cols=cols;

    memset(this->data, 0, cols*rows*sizeof(double));
}

Matrix::Matrix(int rows){
    this->data=new double[rows];
    this->rows=rows;
    this->cols=1;

    memset(this->data, 0, cols*rows*sizeof(double));
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

void Matrix::affine(Matrix &b, Matrix &c){
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //            m, n, k, alpha, A, k, B, n, beta, C, n);
    //A mxk
    //B kxn

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                this->rows, b.cols, this->cols, 1, this->data, this->cols, b.data, b.cols, 0, c.data, b.cols);
}

void Matrix::apply_tanh(){
    vdTanh(rows*cols, this->data, this->data);
//    int total=cols*rows;
//    for (int i=0;i<total;i++){
//        data[i]=tanh(data[i]);
//    }
}

void Matrix::add_scalar(double scalar){
    int total=cols*rows;
    for (int i=0;i<total;i++){
        data[i]+=scalar;
    }
}

void Matrix::apply_sigmoid(){
    int total=cols*rows;
    for (int i=0;i<total;i++){
        data[i]=1.0/(1.0+exp(-data[i]));
    }
}

void Matrix::apply_rectify(){
    int total=cols*rows;
    for (int i=0;i<total;i++){
        if (data[i]<0){
            data[i]=0;
        }
        //data[i]=fmax(data[i],0);
    }
}

void Matrix::fast_copy(Matrix &b){
    memcpy(this->data, b.data, rows*cols*sizeof(double));
}


void Matrix::load_from_file (std::ifstream &f){
    std::string line;
    std::getline(f, line);
    printf("%d %d\t%s\n",rows, cols, line.c_str());
    std::getline(f, line);
    std::istringstream in(line.c_str());
    for (int col=0;col<cols;col++){
        for (int row=0;row<rows;row++){
            int i=row*cols+col;
            in>>data[i];
        }
    }
}

void Matrix::add(Matrix &b, Matrix &rezult){
    MKL_Domatadd ('r', 'n', 'n', rows, cols, 1.0, b.data, cols, 1.0, this->data, cols, rezult.data, cols);
//    int total=rows*cols;
//    for (int i=0;i<total;i++){
//        rezult.data[i]=this->data[i]+b.data[i];
//    }
}

void Matrix::cmultiply(Matrix &b, Matrix &rezult){
    int total=rows*cols;
    for (int i=0;i<total;i++){
        rezult.data[i]=this->data[i]*b.data[i];
    }
}

void Matrix::copy(Matrix &b){
    memcpy(this->data, b.data, rows*cols*sizeof(double));
}

void Matrix::multiply(Matrix &b, Matrix &c){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                this->rows, b.cols, this->cols, 1, this->data, this->cols, b.data, b.cols, 0, c.data, b.cols);
//    rezult.reset();
//    int index_rez=0;
//    for (int row=0;row<this->rows;row++){
//        for (int col=0;col<b.cols;col++){
//            for (int ii=0;ii<b.rows;ii++){
//                int index_a=row*this->cols+ii;
//                int index_b=ii*b.cols+col;
//                rezult.data[index_rez]+=this->data[index_a]*b.data[index_b];
//            }
//            index_rez++;
//        }
//    }
}

void Matrix::print(){
    printf("\n\nMatrix %d %d\n", rows, cols);
    int index=0;
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            printf("%f\t", this->data[index]);
            index++;
        }
        if (cols!=1){
            printf("\n");
        }
    }
}

void Matrix::reset(){
    memset(this->data, 0, rows*cols*sizeof(double));
}

LSTM::LSTM(const LSTM &copy){
    printf("\n\n\n\nHEEEEELPPP!!!!!!!\n\n\n\n\n");
}

LSTM::LSTM(){
}

LSTM::~LSTM(){
    //prevent double free
    this->i_ait.data=0x0;
    this->i_aot.data=0x0;
    this->i_aft.data=0x0;
    this->i_agt.data=0x0;
}

void LSTM::reset(){
    this->ct.reset();
    this->ht.reset();
}

void LSTM::add_input(Matrix &input){
    //tmp = affine_transform({vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1});
    //exit(0);
    this->i_ait.data=tmp.data;
    this->i_aft.data=tmp.data+hidden_size;
    this->i_aot.data=tmp.data+2*hidden_size;
    this->i_agt.data=tmp.data+3*hidden_size;

    this->p_x2i.multiply(input, tmp);
    this->p_h2i.multiply(this->ht,tmp2);
    tmp.add(tmp2, tmp);
    tmp.add(this->p_bi, tmp);


    this->i_ait.apply_sigmoid();
    this->i_aft.add_scalar(1.0);//add forget bias
    this->i_aft.apply_sigmoid();
    this->i_aot.apply_sigmoid();
    this->i_agt.apply_tanh();

    this->ct.cmultiply(this->i_aft, ct);
    this->i_ait.cmultiply(i_agt, i_ait);
    ct.add(this->i_ait, ct);
    ht.fast_copy(ct);
    ht.apply_tanh();
    ht.cmultiply(i_aot, ht);
}

LSTM::LSTM(int input_size, int hidden_size){

    this->input_size=input_size;
    this->hidden_size=hidden_size;

    this->ct=Matrix(hidden_size);
    this->ht=Matrix(hidden_size);
    this->p_x2i=Matrix(hidden_size*4, input_size);
    this->p_h2i=Matrix(hidden_size*4, hidden_size);
    this->m_x2i=Matrix(hidden_size*4, input_size);
    this->m_h2i=Matrix(hidden_size*4, hidden_size);
    this->p_bi=Matrix(hidden_size*4);

    this->tmp=Matrix(hidden_size*4);
    this->tmp2=Matrix(hidden_size*4);
    this->i_ait.data=tmp.data;
    this->i_aft.data=&tmp.data[hidden_size];
    this->i_aot.data=&tmp.data[2*hidden_size];
    this->i_agt.data=&tmp.data[3*hidden_size];
    //memory trick to avoid allocation and memcpy
    this->i_ait.rows=hidden_size; this->i_ait.cols=1;
    this->i_aft.rows=hidden_size; this->i_aft.cols=1;
    this->i_aot.rows=hidden_size; this->i_aot.cols=1;
    this->i_agt.rows=hidden_size; this->i_agt.cols=1;

}

void LSTM::load_from_file(std::ifstream &f){
    p_x2i.load_from_file(f);
    p_h2i.load_from_file(f);
    m_x2i.load_from_file(f);
    m_h2i.load_from_file(f);
    //apply the masks
    m_x2i.cmultiply(p_x2i, p_x2i);
    m_h2i.cmultiply(p_h2i, p_h2i);
    p_bi.load_from_file(f);
}