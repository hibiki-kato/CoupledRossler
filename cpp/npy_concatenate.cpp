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
#include "shared/matplotlibcpp.h"
#include "cnpy/cnpy.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;

int main(){
    const char* a_name = "../generated_lam/sync_gen_laminar_epsilon0.038_a0.165_c10_f0.2_omega0.95-0.99_t100000_1500check100progress10^-16-10^-9perturb.npy";
    const char* b_name = "../generated_lam/sync_gen_laminar_epsilon0.038_a0.165_c10_f0.2_omega0.95-0.99_t2000001500check200progress10^-16-10^-9perturb.npy";
    int check_point = 1e+5; // the last time of a (not equal to time in the file name, usually a nice round number)

    Eigen::MatrixXd a = npy2EigenMat<double>(a_name, true);
    Eigen::MatrixXd b = npy2EigenMat<double>(b_name, true);

    check_point *= 100; //when dt = 0.01
    Eigen::MatrixXd c(a.rows(), check_point + b.cols());
    c.leftCols(check_point) = a.leftCols(check_point);
    c.rightCols(b.cols()) = b;
    
     plt::figure_size(780, 780);
    // Add graph titlecc
    std::vector<double> x(c.cols()),y(c.cols());
    for(int i=0;i<c.cols();i++){
        x[i]=c(6, i);
        y[i]=i;
    }

    plt::plot(x,y);
    plt::save("../../test_concatenate.png");
    std::cout << "Succeed?" << std::endl;
    char none;
    std::cin >> none;
    std::string bname_str = std::string(b_name);
    bname_str = "../" + bname_str;
    EigenMat2npy(c,bname_str);

}