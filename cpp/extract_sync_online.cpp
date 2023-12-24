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
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_next(CoupledRossler& CR, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXd previous);
int shift(double pre_theta, double theta, int rotation_number);
bool isSync(double a, double b, double sync_criteria, double center);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+7;
    double dump = 0;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.035;
    int epsilon_num = 96;
    Eigen::VectorXd epsilons = Eigen::VectorXd::LinSpaced(epsilon_num, 0.03, 0.05);
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    double sync_criteria = 0.8;
    double d = 1.2; //  if phase_diff is in 2πk + d ± sync_criteria then it is synchronized
    int numThreads = omp_get_max_threads();
    std::cout << numThreads << " threads" << std::endl;

    int window = 500; // how long the sync part should be. (sec)
    window *= 100; // 100 when dt = 0.01 
    int trim = 250; // how much to trim from both starts and ends of sync part
    trim *= 100; // 100 when dt = 0.01
    int skip = 1; // plot every skip points
    
    int plotDim[2] = {1, 4};
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "20";
    plotSettings["figure.max_open_warning"] = 50; // set max open figures to 50
    plt::rcparams(plotSettings);
    
    int steps = static_cast<int>((t - t_0) / dt + 0.5);
    std::atomic<int> progress(0);
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(CR, sync_criteria, d, steps, epsilons, plotDim, window, trim) shared(progress)
    for (int i = 0; i < epsilon_num; i++) {
        CR.epsilon = epsilons(i);
        Eigen::VectorXd previous = CR.x_0;
        Eigen::VectorXd n = Eigen::VectorXd::Zero(previous.rows());
        Eigen::VectorXd theta(2);
        theta(0) = std::atan2(previous(1), previous(0)); // rotation angle of system1
        theta(1) = std::atan2(previous(4), previous(3));
        std::vector<double> x;
        std::vector<double> y;

        std::vector<double> synced_x;
        std::vector<double> synced_y;
        
        for (int j = 0; j < steps; j++) {
            std::tie(n, theta, previous) = calc_next(CR, n, theta, previous);
            if (isSync(theta(0) + 2*n(0)*M_PI, theta(1) + 2*n(1)*M_PI, sync_criteria, d)){
                x.push_back(previous(plotDim[0]-1));
                y.push_back(previous(plotDim[1]-1));
            }
            else{
                if (x.size() > window){
                    synced_x.insert(synced_x.end(), x.begin()+trim, x.end()-trim);
                    synced_y.insert(synced_y.end(), y.begin()+trim, y.end()-trim);
                }
                x.clear();
                y.clear();
            }
        }
        if (x.size() > window){
            // 前後trim点を削除してinsert
            synced_x.insert(synced_x.end(), x.begin()+trim, x.end()-trim);
            synced_y.insert(synced_y.end(), y.begin()+trim, y.end()-trim);
        }
        x.clear();
        y.clear();

        #pragma omp critical
        {   
            if (synced_x.size() > 0){
                // plot
                plt::figure_size(1000, 1000);
                std::map<std::string, std::string> plotSettings;
                plotSettings["alpha"] = "0.01";
                plt::scatter(synced_x, synced_y, 1);
                std::string xyz = "xyzxyz";
                char xlabel[100];
                char ylabel[100];
                sprintf(xlabel, "$%c_%d$", xyz[plotDim[0]-1], plotDim[0] / 3 + 1);
                sprintf(ylabel, "$%c_%d$", xyz[plotDim[1]-1], plotDim[1] / 3 + 1);
                plt::xlabel(xlabel);
                plt::ylabel(ylabel);

                // save
                std::ostringstream oss;
                oss << "../../sync/sync_epsilon" << CR.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << "_window" << window/100 <<".png";
                std::string plotfname = oss.str(); // 文字列を取得する
                std::cout << "Saving result to " << plotfname << std::endl;
                plt::save(plotfname);
                plt::close();
                synced_x.clear();
                synced_y.clear();
            }
        }
        progress++;
        if (omp_get_thread_num() == 0) {
                std::cout << "\r processing " << progress  << "/" << epsilon_num << std::flush;
        }
    }
    myfunc::duration(start, std::chrono::system_clock::now());
}

double shift(double pre_theta, double theta, double rotation_number){
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



std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_next(CoupledRossler& CR, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXd previous){
    Eigen::VectorXd now = CR.rk4(previous);
    Eigen::VectorXd theta(2);
    theta(0) = std::atan2(now(1), now(0)); // rotation angle of system1
    theta(1) = std::atan2(now(4), now(3));
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}

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