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
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    Eigen::MatrixXcd trajectory = npy2EigenMat();
    int dim = trajectory.rows() - 1;
    std::cout << "dim = " << dim << std::endl;
    int cols = trajectory.cols();
    std::cout << "cols = " << cols << std::endl;
    std::cout << "t=" << std::abs(trajectory(dim,0)) << "to" << std::abs(trajectory(dim, cols-1)) <<std::endl;

    /*
     ██     █                    █                        █     █  █       ██   █               █  █
    ██████  █                    █                       ██     █  █     █████  █               █  █
    ██   █  █          ██   ██                           ███    █  █     █      █               █  █
    ██   █  █   ████  ████ ████  █  ██████   █████      ██ █    █  █     █      ██████   ████   █  █  █████
    ██   █  █  ██   █  █    █    █  ██   █  ██  ██      █  █    █  █     ███    ██   █  █   ██  █  █  █
    █████   █  █    █  █    █    █  █    █  █    █      █  ██   █  █       ███  █    █  █   ██  █  █  ██
    ██      █  █    █  █    █    █  █    █  █    █     ██████   █  █         ██ █    █  ██████  █  █   ███
    ██      █  █    █  █    █    █  █    █  █    █     █    ██  █  █         ██ █    █  █       █  █     ██
    ██      █  ██  ██  ██   ██   █  █    █  ██  ██     █     █  █  █         █  █    █  ██      █  █     ██
    ██      █   ████    ███  ███ █  █    █   █████    █      █  █  █     █████  █    █   ████   █  █  ████
                                                 █
                                                ██
                                            █████
    */
    std::cout << "plotting" << std::endl;
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2000, 3000);
    int skip = 1; // plot every skip points
    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);
    //time
    for(int i=0;i<x.size();i++){
        x[i]=std::abs(trajectory(dim, i*skip));
    }
    //plot
    for(int i=0; i < dim; i+=4){
        for(int j=0; j < y.size(); j++){
            y[j]=std::abs(trajectory(i, j*skip));
        }
        plt::subplot(dim,1, i+1);
        plt::yscale("log");
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
    }
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../test.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

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