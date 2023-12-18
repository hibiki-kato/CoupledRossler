/**
 * @file stagger_and_step.cpp
 * @author Hibiki Kato
 * @brief using stagger and step method to detect chaotic saddle
 * @version 0.1
 * @date 2023-06-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include <omp.h>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec(const char* fname);
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);


int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    // generating laminar sample for detection
    // !DO NOT CHANGE!
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 5000;
    double latter = 20;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_10000period_dt0.01eps0.005.npy");

    //generating laminar sample for detection
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, x_0);

    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numCols = laminar.cols() / 10;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numCols);
    for (int i = 0; i < numCols; i++){
        int colIdx = 10 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }

    beta = 0.417;
    nu = 0.00018;
    latter = 1;
    t = 10000;
    dt = 0.01;
    t_0 = 0;
    // Eigen::MatrixXcd loaded = npy2EigenMat("../../generated_lam/generated_laminar_beta_0.416nu_0.00018_dt0.01_38100period500check20progresseps0.02.npy");
    // x_0 = loaded.block(0, t_0*100 - 1, 14, 1);
    x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_13348period_dt0.01eps0.005.npy");
    double epsilon=1E-1; // 4~5E-2 is appropriate
    int skip = 1000;


    double check_sec = 2000;
    double progress_sec = 10;
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;

    LongLaminar LL(nu, beta, f, dt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, check_sec, progress_sec, threads);
    Eigen::MatrixXcd calced_laminar = LL.stagger_and_step_();
    plt::figure_size(1200, 780);
    // Add graph title
    std::vector<double> x(calced_laminar.cols()),y(calced_laminar.cols());
    int reach = static_cast<int>(calced_laminar.bottomRightCorner(1, 1).cwiseAbs()(0, 0) + 0.5); 

    for(int i=0;i<calced_laminar.cols();i++){
        x[i]=calced_laminar.cwiseAbs()(3, i);
        y[i]=calced_laminar.cwiseAbs()(4, i);
    }

    plt::plot(x,y);
    std::ostringstream oss;
    oss << "../../generated_lam_imag/generated_laminar_beta_" << beta << "nu_" << nu <<"_"<< reach << "period.png";  // 文字列を結合する
    std::string filename = oss.str(); // 文字列を取得する
    std::cout << "\n Saving result to " << filename << std::endl;
    plt::save(filename);

    oss.str("");
    oss << "../../generated_lam/generated_laminar_beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << reach << "period" << check_sec << "check" << progress_sec << "progress" << "eps" << epsilon << ".npy";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    EigenMt2npy(calced_laminar, fname);

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
}

Eigen::VectorXcd npy2EigenVec(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    std::complex<double>* data = arr.data<std::complex<double>>();
    Eigen::Map<Eigen::VectorXcd> vec(data, arr.shape[0]);
    return vec;
}

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)){
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}