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
#include <chrono> 
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_next(CoupledRossler& CR, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXd previous);
int shift(double pre_theta, double theta, int rotation_number);
bool isSync(double a, double b, double sync_criteria, double center);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+9;
    double dump = 1e+4;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.035;
    int epsilon_num = 32;
    // Eigen::VectorXd epsilons = Eigen::VectorXd::LinSpaced(epsilon_num, 0.037, 0.0416);
    // 0.416から指数的に減少
    Eigen::VectorXd epsilons = Eigen::VectorXd::LinSpaced(epsilon_num, -2.9, -4);
    for (int i = 0; i < epsilon_num; i++) epsilons(i) = 0.0416 - std::pow(10, epsilons(i));
    std::cout << epsilons << std::endl;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/chaotic.npy", true);
    double sync_criteria = 1.2;
    double d = 1.2; //  if phase_diff is in 2πk + d ± sync_criteria then it is synchronized
    int numThreads = omp_get_max_threads();

    int window = 500; // how long the sync part should be. (sec)
    int trim = 250; // how much to trim from both starts and ends of sync part
    int skip = 100; // period of checking sync(step)
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    std::vector<double> average_durations(epsilon_num);
    int progress = 0;
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(CR, sync_criteria, d, epsilons, window, trim, skip) shared(progress, average_durations, epsilon_num)
    for (int i = 0; i < epsilon_num; i++) {
        CR.epsilon = epsilons(i);
        Eigen::VectorXd previous = CR.x_0;
        Eigen::VectorXd n = Eigen::VectorXd::Zero(previous.rows());
        Eigen::VectorXd theta(2);
        theta(0) = std::atan2(previous(1), previous(0)); // rotation angle of system1
        theta(1) = std::atan2(previous(4), previous(3));
        double duration;
        std::vector<double> durations;
        // dump
        for (int j = 0; j < CR.dump_steps; j++) {
            std::tie(n, theta, previous) = calc_next(CR, n, theta, previous);
        }
        
        for (long long j = 0; j < CR.steps; j++) {
            std::tie(n, theta, previous) = calc_next(CR, n, theta, previous);
            if(j % skip == 0){
                if (isSync(theta(0) + 2*n(0)*M_PI, theta(1) + 2*n(1)*M_PI, sync_criteria, d)){
                    duration += dt*skip;
                }else{
                    if (duration - trim*2 > window){
                        durations.push_back(duration-trim*2);
                    }
                    duration = 0;
                }
            }
        }
        if (duration-trim*2 > window){
            durations.push_back(duration-trim*2);
        }
        #pragma omp atomic
        progress++;
        #pragma omp critical
        std::cout << "\r processing " << progress  << "/" << epsilon_num << std::flush;
        average_durations[i] = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
        // //durationsを保存
        // std::ostringstream oss;
        // std::ofstream ofs("durations.txt");
        // for (int j = 0; j < durations.size(); i++){
        //     ofs << durations[j] << std::endl;
        // }
        // ofs.close();
    }
    // epsilonsとaverage_durationsを保存
    std::ostringstream oss;
    oss << "../../average_durations/data/epsilon" << epsilons(0) << "-" << epsilons(epsilon_num-1) << "_" << epsilon_num << "num" << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_window" << window <<".txt";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < epsilon_num; i++){
        ofs << epsilons(i) << " " << average_durations[i] << std::endl;
    }
    ofs.close();
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "20";
    plt::rcparams(plotSettings);

    Eigen::VectorXd epsilons_diff = 0.0416 - epsilons.array();
    std::vector<double> epsilons_vec(epsilons_diff.data(), epsilons_diff.data() + epsilons_diff.size());

    // plot
    plt::figure_size(1200, 800);
    plt::scatter(epsilons_vec, average_durations, 5);
    plt::yscale("log");
    plt::xscale("log");
    plt::xlabel("epsilon");
    plt::ylabel("average laminar duration");

    // save
    oss.str("");
    oss << "../../average_durations/epsilon" << epsilons(0) << "-" << epsilons(epsilon_num-1) << "_" << epsilon_num << "num" << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_window" << window <<".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
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