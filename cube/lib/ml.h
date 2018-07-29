#pragma once
typedef class Matrix{
    private:
    int cols;
    int rows;
    float **data;
    public:
    Matrix (int cols, int rows);
    ~Matrix();
    
};