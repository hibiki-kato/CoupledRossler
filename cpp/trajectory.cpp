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
    double dump = 1e+4;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.03;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    // Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/chaotic.npy", true);

    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXd trajectory = CR.get_trajectory();

    int plot_dim1 = 1;
    int plot_dim2 = 5;
    int skip = 1; // plot every skip points
    // /*
    //         █
    // █████   █          █    █    █
    // █    █  █          █    █
    // █    █  █   ████  ████ ████  █  █ ███    █████
    // █   ██  █  ██  ██  █    █    █  ██  ██  █   █
    // █████   █  █    █  █    █    █  █    █  █   █
    // █       █  █    █  █    █    █  █    █  █   █
    // █       █  █    █  █    █    █  █    █  ████
    // █       █  ██  ██  █    █    █  █    █  █
    // █       █   ████    ██   ██  █  █    █  █████
    //                                        ██   ██
    //                                        █    ██
    //                                         █████
    // */
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1000, 1000);
    // Add graph title
    std::vector<double> x(trajectory.cols()/skip),y(trajectory.cols()/skip);
    for(int i=0;i<x.size();i++){
        x[i]=trajectory(plot_dim1 - 1, i*skip);
        y[i]=trajectory(plot_dim2 - 1, i*skip);
    }
    std::map<std::string, std::string> keywords;
    keywords["lw"] = "1";
    // plt::plot(x,y, keywords);
    // plt::xlim(-17, 20);
    // plt::ylim(-17, 20);
    plt::scatter(x,y, 1);
    std::ostringstream oss;
    oss << "../../traj_img/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << ".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);


    /*
     ████                        ███
    █                           █   █
    █       ███  █    █   ███       █  █ ███   █ ███  █    █
    ██         █ ██   █  ██  █      █  ██  ██  ██  ██  █   █
     ███       █  █  ██  █   ██    ██  █    █  █    █  █  ██
        █   ████  █  █   ██████   ██   █    █  █    █  ██ █
        ██ █   █   █ █   █       ██    █    █  █    █   █ █
        █  █   █   ██    ██     ██     █    █  ██  ██   ██
    ████   █████   ██     ████  ██████ █    █  █ ███     █
                                               █         █
                                               █        █
                                               █      ███
    */
    oss.str("");
    //  文字列を取得する
    oss << "../../traj/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << ".npy";
    std::string npyfname = oss.str();
    std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMat2npy(trajectory, npyfname);
    
    myfunc::duration(start, std::chrono::system_clock::now());
}