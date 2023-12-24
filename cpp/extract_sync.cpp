/**
 * @file extract_sync.cpp
 * @author Hibiki Kato
 * @brief extract synchronized part of trajectory
 * @version 0.1
 * @date 2023-09-19
 *
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
#include "Eigen_numpy_converter.hpp"
#include "myFunc.hpp"

namespace plt = matplotlibcpp;
int shift(double pre_theta, double theta, int rotation_number);
bool isSync(double a, double b, double sync_criteria, double center);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 0;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.035;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    double sync_criteria = 0.8;
    double d = 1.2; //  if phase_diff is in 2πk + d ± sync_criteria then it is synchronized

    int window = 500; // how long the sync part should be. (sec)
    window *= 100; // 100 when dt = 0.01 
    int trim = 250; // how much to trim from both starts and ends of sync part
    trim *= 100; // 100 when dt = 0.01
    int skip = 1; // plot every skip points
    int numthreads = omp_get_max_threads();
    int plotDim1 = 1;
    int plotDim2 = 4;

    std::cout << "calculating trajectory" << std::endl;
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXd trajectory = CR.get_trajectory(); //wide matrix
    Eigen::MatrixXd angles = Eigen::MatrixXd::Zero(trajectory.cols(), 2);
    for(int i=0; i < trajectory.cols(); i++){
        angles(i, 0) = std::atan2(trajectory(1, i), trajectory(0, i));
        angles(i, 1) = std::atan2(trajectory(4, i), trajectory(3, i));
    }
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

    std::cout << "extracting sync" << std::endl;
    std::vector<std::vector<double>> synced;
    synced.resize(trajectory.rows());
    int counter = 0;
    for (int i = 0; i < angles.rows(); i++){
        if (isSync(angles(i, 0), angles(i, 1), sync_criteria, d)){
            counter++;
        }
        else{
            if (counter >= window){
                //adding synchronized part to synced
                for (int j = 0 + trim; j < counter - 1 - trim; j++){
                    for (int k = 0; k < trajectory.rows(); k++){
                        synced[k].push_back(trajectory(k, j + i - counter));
                    }
                }
            }
            counter = 0;
            }
    }
    //adding last part to synced
    if (counter >= window){
        for (int j = 0 + trim; j < counter - 1 - trim; j++){
            for (int k = 0; k < trajectory.rows(); k++){
                synced[k].push_back(trajectory(k, j + angles.rows() - counter));
            }
        }
    }
    /*
            █             
    █████   █          █  
    ██  ██  █          █  
    ██   █  █   ████  ████
    ██  ██  █  ██  ██  █  
    █████   █  █    █  █  
    ██      █  █    █  █  
    ██      █  █    █  █  
    ██      █  ██  ██  ██ 
    ██      █   ████    ██
    */
    std::cout << synced[0].size() << "/" << angles.rows() << " is synchronized" <<std::endl;
    std::cout << "plotting" << std::endl;
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "15";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1000, 1000);
    
    std::map<std::string, double> keywords;
    std::vector<double> x(synced[0].size()/skip),y(synced[0].size()/skip);
    for (int i = 0; i < x.size(); i++){
        x[i] = synced[plotDim1-1][i*skip];
        y[i] = synced[plotDim2-1][i*skip];
    }
    // plt::xlabel("$t [sec]$");
    // plt::ylabel("$|U_{1}|$");
    plt::scatter(x, y);
    std::ostringstream oss;
    oss << "../../sync/sync_epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << "_window" << window <<".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    /* 
     ████                          █
    █                              █
    █       ███  █    █   ███     ████   ████         █ ███   █ ███  █    █
    ██         █  █   █  ██  █     █    ██  ██        ██  ██  ██  ██  █   █
     ███       █  █  ██  █   ██    █    █    █        █    █  █    █  █  ██
        █   ████  ██ █   ██████    █    █    █        █    █  █    █  ██ █
        ██ █   █   █ █   █         █    █    █        █    █  █    █   █ █
        █  █   █   ███   ██        ██   ██  ██     █  █    █  ██  ██   ███
    ████   █████   ██     ████      ██   ████      ██ █    █  █████     █
                                                              █         █
                                                              █        █
                                                              █      ███
    */
        // reset oss
    oss.str("");
    oss << "../../sync/npy/sync_epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << "_window" << window/100 <<".npy";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    Eigen::MatrixXd matrix(synced.size(), synced[0].size());
    for (int i = 0; i < synced.size(); i++) {
        for (int j = 0; j < synced[0].size(); j++) {
            matrix(i, j) = synced[i][j];
        }
    }
    EigenMat2npy(matrix, fname);

    myfunc::duration(start, std::chrono::system_clock::now());
}

int shift(double pre_theta, double theta, int rotation_number){
    //forward
    if ((theta - pre_theta) < -M_PI){
        rotation_number += 1;
    }
    //backward
    else if ((theta - pre_theta) > M_PI){
        rotation_number -= 1;
    }

    return rotation_number;
}

/**
 * @brief given 2 angles, check if they are in sync
 * 
 * @param a : angle 1
 * @param b  : angle 2
 * @param epsilon : tolerance
 * @return true : sync
 * @return false : not sync
 */
bool isSync(double a, double b, double sync_criteria, double center) {
    double lowerBound = center - sync_criteria;
    double upperBound = center + sync_criteria;
    int n = 0;
    double diff = std::abs(a - b);
    // std::cout << diff << std::endl;
    while (lowerBound <= diff) {
        if (lowerBound <= diff && diff <= upperBound) {
            return true;
        }
        n++;
        lowerBound += 2  * M_PI;
        upperBound += 2  * M_PI;
    }
    return false;
}