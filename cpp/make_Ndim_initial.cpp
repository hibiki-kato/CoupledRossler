#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenVec2npy(Eigen::VectorXd Vec, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::VectorXcd perturbation(Eigen::VectorXcd state,  std::vector<int> dim, int s_min = -1, int s_max = -1);
void EigenVec2npy(Eigen::VectorXcd Vec, std::string fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    double nu = 0.00018;
    double beta = 0.41525 ;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 100000;
    double latter = 1;
    int dim = 14;

    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.415_nu0.00018_100000period_dt0.01.npy");
    std::vector<std::complex<double>> x_0_vec(x_0.data(), x_0.data() + x_0.size());
    for(int i = 0; i < dim - x_0.size(); i++){
        x_0_vec.push_back(std::complex<double>(1e-6, 1e-6));
    }
    x_0 = Eigen::Map<Eigen::VectorXcd>(x_0_vec.data(), x_0_vec.size());
    std::cout << x_0 << std::endl;
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    std::ostringstream oss;
    Eigen::VectorXcd result = SM.get_trajectory_().topRightCorner(dim, 1);

    std::cout << result << std::endl;
    oss << "../../initials/beta" << beta << "_nu" << nu<< "_" << dim << "dim_period.npy";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    EigenVec2npy(result, fname);

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

void EigenMt2npy(Eigen::MatrixXd Mat, std::string fname){
    Eigen::MatrixXd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to npy file
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

void EigenVec2npy(Eigen::VectorXcd Vec, std::string fname){
    std::vector<std::complex<double>> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}