/**
 * @file Eigen_numpy_converter.hpp
 * @author hibiki kato
 * @brief Convert Eigen to numpy and vice versa
 * @version 0.1
 * @date 2023-10-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include "cnpy/cnpy.h"
#include <complex>
// #include <type_traits>
#include <string>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> npy2EigenVec(const char* fname){
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    T* loaded_data = arr.data<T>();
    int size = arr.shape[0];
    Eigen::Matrix<T, Eigen::Dynamic, 1> Vec(size);
    for(int i = 0; i < size; i++){
        Vec(i) = loaded_data[i];
    }
    return Vec;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> npy2EigenMat(const char* fname){
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    T* loaded_data = arr.data<T>();
    int size = arr.shape[0];
    int dim = arr.shape[1];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mat(size, dim);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < dim; j++){
            Mat(i, j) = loaded_data[i*dim + j];
        }
    }
    return Mat;
}

template <typename Vector>
void EigenVec2npy(Vector Vec, std::string fname){
    long unsigned int size = Vec.size();
    std::vector<typename Vector::Scalar> data(2*size);
    for(int i = 0; i < size; i++){
        data[i] = Vec(i);
    }
    cnpy::npy_save(fname, &data[0], {size}, "w");
}

template <typename Matrix>
void EigenMat2npy(Matrix Mat, std::string fname){
    Matrix transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Matrix> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}