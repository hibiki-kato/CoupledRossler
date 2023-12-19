/**
 * @file phase.cpp
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
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "Eigen_numpy_converter.hpp"
#include "myFunc.hpp"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 1e+2;
    double omega1 = 0.95;
    double omega2 = 0.99;
    double epsilon = 0.041;
    double a = 0.165;
    double c = 10;
    double f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array() + 1) / 2;
    int numthreads = omp_get_max_threads();
    CoupledRossler CR(omega1, omega2, epsilon, a, c, f, dt, t_0, t, dump, x_0);
    Eigen::MatrixXd trajectory = CR.get_trajectory();
    /*                                                                        
                  ██                                                                                                    
        ██████    ██                                                ██                                                  
        ██   ██   ██             █             █                     █                    █                             
        ██    ██  ██             █             █                                          █                             
        ██    ██  ██    ████   █████         █████  ██ ██   █████   ██    ████     ████ █████    ████    ██ ██ ██    ██ 
        ██    ██  ██   ██  ██   ██            ██    ████    █  ██   ██   ██  ██   ██  █  ██     ██  ██   ████   █    ██ 
        ██   ██   ██  ██    ██  ██            ██    ██          ██  ██  ██   ██  ██      ██    ██    ██  ██     █    █  
        ██████    ██  ██    ██  ██            ██    ██          ██  ██  ██    █  ██      ██    ██    ██  ██     ██  ██  
        ██        ██  █     ██  ██            ██    ██      ██████  ██  ███████  █       ██    █     ██  ██      █  ██  
        ██        ██  ██    ██  ██            ██    ██     ██   ██  ██  █        ██      ██    ██    ██  ██      ██ █   
        ██        ██  ██    ██  ██            ██    ██     █    ██  ██  ██       ██      ██    ██    ██  ██      ████   
        ██        ██   ██  ██    ██            ██   ██     ██  ███  ██   ██   █   ██  █   ██    ██  ██   ██       ███   
        ██        ██    ████     ███           ███  ██      ██████  ██    █████    ████   ███    ████    ██       ██    
                                                                    ██                                            ██    
                                                                    ██                                            █     
                                                                    ██                                           ██     
                                                                  ███                                          ███      
    */
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 10; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "15";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(3600, 1200);
    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);
    // times for x axis
    for(int i=0;i<x.size();i++){
        x[i]=trajectory(trajectory.rows()-1, i*skip);
    }
    // plot trajectory
    for(int i=0; i < 2; i++){
        for(int j=0; j < y.size(); j++){
            y[j]=trajectory(i*3, j*skip);
        }
        plt::subplot(2, 2, i*2+1);
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$x_{" + std::to_string(i+1) + "}$");
    }

    /*
                            █
       ██                   █
       ██                   █
       █ █   █ ███    █████ █   ███
      █  █   ██  ██  █   █  █  ██  █
      █  █   █    █  █   █  █  █   ██
     ██████  █    █  █   █  █  ██████
     █    █  █    █  ████   █  █
     █    ██ █    █  █      █  ██
    █      █ █    █  █████  █   ████
                    ██   ██
                    █    ██
                     █████
    */
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
              ██                                                       ██                 
    ██████    ██                                                       ██                 
    ██   ██   ██             █                                         ██                 
    ██    ██  ██             █                                         ██                 
    ██    ██  ██    ████   █████           █████   ██ ███      ███ ██  ██    ████    ████ 
    ██    ██  ██   ██  ██   ██             █  ██   ███  ██    ██  ███  ██   ██  ██  ██  █ 
    ██   ██   ██  ██    ██  ██                 ██  ██    ██  ██    ██  ██  ██   ██  █     
    ██████    ██  ██    ██  ██                 ██  ██    ██  ██    ██  ██  ██    █  ███   
    ██        ██  █     ██  ██             ██████  ██    ██  █     ██  ██  ███████   ████ 
    ██        ██  ██    ██  ██            ██   ██  ██    ██  ██    ██  ██  █            ██
    ██        ██  ██    ██  ██            █    ██  ██    ██  ██    ██  ██  ██           ██
    ██        ██   ██  ██    ██           ██  ███  ██    ██   ██  ███  ██   ██   █  █   ██
    ██        ██    ████     ███           ██████  ██    ██    ███ ██  ██    █████  █████ 
                                                                   █                      
                                                                   █                      
                                                              █   ██                      
                                                              █████                       
    */
    std::cout << "plotting angles" << std::endl;
    for(int i=0; i < angles.cols(); i++){
        for(int j=0; j < y.size(); j++){
            y[j]=angles(j*skip, i);
        }
        plt::subplot(2, 2, 2*i+2);
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
        
    }
    
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.6)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.4)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../phase/epsilon" << epsilon << "_a" << a << "_c" << c << "_f" << f << "_dt" << dt << "_t" << t << "_dump" << dump << "_omega(" << omega1 << "," << omega2 << ").png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start, std::chrono::system_clock::now());
}