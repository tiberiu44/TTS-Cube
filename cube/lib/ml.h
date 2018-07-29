#pragma once
#include <fstream>
#include <vector>
#include <memory.h>

class Matrix{
    private:
    int cols;
    int rows;
    double *data;
    public:
    Matrix();
    Matrix(const Matrix &copy);
    Matrix (int rows, int cols);
    Matrix (int rows);
    ~Matrix();
    Matrix multiply(Matrix &b);
    void multiply(Matrix &b, Matrix &rezult);
    void load_from_file(std::ifstream&);

    Matrix& operator=(const Matrix &other){
        this->data=new double[other.cols*other.rows];
        this->cols=other.cols;
        this->rows=other.rows;
        memcpy(this->data, other.data, cols*rows*sizeof(double));
        return *this;
    }

};

class LSTM{
    private:
        //i
        Matrix p_x2i;
        Matrix p_h2i;
        Matrix p_bi;
        //o
        Matrix p_gh;
        Matrix p_bh;
        Matrix p_gx;
        Matrix p_bx;
        Matrix p_gc;
        Matrix p_bc;
        //c
        Matrix p_x2c;
        Matrix p_h2c;
        Matrix p_bc;
        //hidden
        Matrix h;

    public:
        public LSTM()
}