/**
 * @file npy_concatenate.cpp
 * @author Hibiki Kato
 * @brief 2 npy files concatenation. This code basically used for concatenating generated laminar.
 * @version 0.1
 * @date 2023-06-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <complex>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
#include "Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;

int main(){
    const char* a_name = "../../generated_lam/sync_gen_laminar_beta_0.417nu_0.00018_dt0.01_50000period5000check500progress10^-14-10^-5perturb_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy";
    const char* b_name = "../../generated_lam/sync_gen_laminar_beta_0.417nu_0.00018_dt0.01_50000period6000check500progress10^-10-10^-7perturb_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy";
    int check_point = 30000; // the last time of a (not equal to time in the file name, usually a nice round number)

    Eigen::MatrixXcd a = npy2EigenMat<std::complex<double>>(a_name);
    Eigen::MatrixXcd b = npy2EigenMat<std::complex<double>>(b_name);

    check_point *= 100; //when dt = 0.01
    Eigen::MatrixXcd c(a.rows(), check_point + b.cols());
    c.leftCols(check_point) = a.leftCols(check_point);
    c.rightCols(b.cols()) = b;
    
     plt::figure_size(780, 780);
    // Add graph titlecc
    std::vector<double> x(c.cols()),y(c.cols());
    for(int i=0;i<c.cols();i++){
        x[i]=c.cwiseAbs()(14, i);
        y[i]=i;
    }

    plt::plot(x,y);
    plt::save("../../test_concatenate.png");
    std::cout << "Succeed?" << std::endl;
    char none;
    std::cin >> none;
    EigenMat2npy(c, b_name);

}