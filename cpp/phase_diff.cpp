/**
 * @file phase_diff.cpp
 * @author Hibiki Kato
 * @brief 
 * @version 0.1
 * @date 2023-12-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <chrono>
#include "shared/Flow.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+4;
    double dump = 1e+4;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.039;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    int numthreads = omp_get_max_threads();
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);

    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXd trajectory = CR.get_trajectory(); //wide matrix
    // Eigen::MatrixXd trajectory = npy2EigenMat<std::complex<double>>("../../generated_lam/sync_gen_laminar_beta_0.43nu_0.00018_dt0.01_5000period3000check100progress10^-8-10^-5perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy"); //wide matrix
    // Eigen::MatrixXd trajectory = trajectory_.leftCols(500000);

    // trajectoryの1行目をx, 2行目をyとした時の偏角を計算
    Eigen::MatrixXd angles = Eigen::MatrixXd::Zero(trajectory.cols(), 2);
    for(int i=0; i < trajectory.cols(); i++){
        angles(i, 0) = std::atan2(trajectory(1, i), trajectory(0, i));
        angles(i, 1) = std::atan2(trajectory(4, i), trajectory(3, i));
    }

    std::cout << "unwrapping angles" << std::endl;

    //unwrap
    #pragma omp parallel for num_threads(numthreads)
    for (int i = 0; i < angles.cols(); i++){
        int rotation_number = 0;
        for (int j = 0; j < angles.rows(); j++){
            if (j == 0){
                continue;
            }
            //　unwrapされた角度と回転数を返す
            int  n= myfunc::shift(angles(j-1, i), angles(j, i), rotation_number);
            // 一個前の角度に回転数を加える
            angles(j-1, i) += rotation_number * 2 * M_PI;
            // 回転数を更新
            rotation_number = n;
        }
        // 一番最後の角度に回転数を加える
        angles(angles.rows()-1, i) += rotation_number * 2 * M_PI;
    }


    /*
            █                       █  █   ███ ███
    █████   █          █            █      █   █  
    ██  ██  █          █            █     ██  ██  
    ██   █  █   ████  ████      █████  █ ████████ 
    ██  ██  █  ██  ██  █       ██  ██  █  ██  ██  
    █████   █  █    █  █       █    █  █  ██  ██  
    ██      █  █    █  █       █    █  █  ██  ██  
    ██      █  █    █  █       █    █  █  ██  ██  
    ██      █  ██  ██  ██      ██  ██  █  ██  ██  
    ██      █   ████    ██      ███ █  █  ██  ██  
    */

    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 100; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "15";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2400, 800);

    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);

    // times for x axis
    for(int i=0;i<x.size();i++){
        x[i]=trajectory(trajectory.rows()-1, i*skip);
    }

    //plot phase difference
    Eigen::VectorXd diff = (angles.col(0) - angles.col(1)).cwiseAbs();
    for (int i = 0; i < y.size(); i++){
        y[i] = diff(i*skip);
    }
    double x_max = diff.maxCoeff();
    double y_max = diff.maxCoeff();
    for (int i = 0; i < y_max / M_PI / 2; i++){
        plt::axhline(i*2*M_PI, 0.0, x_max, {{"color", "gray"}, {"linestyle", ":"}});
    }
    plt::plot(x,y);
    plt::xlabel("t [sec]");
    plt::ylabel("$|\\psi_{1,2}|$");

    std::ostringstream oss;
    oss << "../../phase_diff/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << ".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start); // 計測終了時間
}