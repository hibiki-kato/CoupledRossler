/**
 * @file stagger_and_step_sync.cpp
 * @author Hibiki Kato
 * @brief stagger and step method using synchronization
 * @version 0.1
 * @date 2023-12-16
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
#include <random>
#include "shared/myFunc.hpp"
#include "shared/Flow.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
bool isLaminar(Eigen::VectorXd phases, double sync_criteria, double center);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_next(CoupledRossler& CR, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXd previous);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 0;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.039;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::MatrixXd loaded = npy2EigenMat<double>("../generated_lam/sync_gen_laminar_epsilon0.038_a0.165_c10_f0.2_omega0.95-0.99_t100000_1500check100progress10^-16-10^-9perturb.npy", true);
    Eigen::VectorXd x_0 = loaded.block(0, t_0*100, 6, 1);
    // Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/epsilon0.038_a0.165_c10_f0.2_omega0.95-0.99_t1500.npy", true);
    // Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;

    double check = 1500;
    double progress = 200;
    int perturb_min = -15;
    int perturb_max = -8;
    int limit = 1e+6; //limitation of trial of stagger and step
    double sync_criteria = 0.8; 
    double d = 1.2; //  if phase_diff is in 2πk + d ± sync_criteria then it is synchronized

    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    int numThreads = omp_get_max_threads();
    int num_variables = 6;
    /*
      ██                                                                    █
    █████                                                                   █
    █      ██                                                               █            ██
    █     ████   ████   █████   █████   ████   █ ██      ████  ██████   █████     █████ ████   ████   █████
    ███    █        █  ██  ██  ██  ██  █   ██  ██           █  ██   █  ██  ██     █      █    █   ██  ██   █
      ███  █        █  █    █  █    █  █   ██  █            █  █    █  █    █     ██     █    █   ██  █    █
        ██ █    █████  █    █  █    █  ██████  █        █████  █    █  █    █      ███   █    ██████  █    █
        ██ █    █   █  █    █  █    █  █       █        █   █  █    █  █    █        ██  █    █       █    █
        █  ██   █   █  ██  ██  ██  ██  ██      █        █   █  █    █  ██  ██        ██  ██   ██      ██   █
    █████   ███ █████   █████   █████   ████   █        █████  █    █   ███ █     ████    ███  ████   █████
                            █       █                                                                █
                           ██      ██                                                                █
                       █████   █████                                                                 █
    */
    Eigen::MatrixXd calced_laminar(num_variables+1, CR.steps+1);
    int stagger_and_step_num = static_cast<int>((t-t_0) / progress + 0.5);
    int check_steps = static_cast<int>(check / dt + 0.5);
    int progress_steps = static_cast<int>(progress / dt + 0.5);
    CR.steps = check_steps; // for what??
    Eigen::VectorXd n = Eigen::VectorXd::Zero(2);

    Eigen::VectorXd next_n(2); // candidate of next n
    double max_perturbation = 0; // max perturbation
    double min_perturbation = 1; // min perturbation

    for (int i; i < stagger_and_step_num; i++){
        std::cout << "\r 現在" << CR.t_0 << "時間" << std::flush;
        bool laminar = true; // flag
        double now_time = CR.t_0;
        
        Eigen::VectorXd now = CR.x_0; // initial state
        Eigen::MatrixXd trajectory = Eigen::MatrixXd::Zero(num_variables+1, progress_steps+1); //wide matrix for progress
        trajectory.topLeftCorner(num_variables, 1) = now;
        trajectory(now.rows(), 0) = now_time;
        Eigen::VectorXd theta(2);
        theta(0) = std::atan2(now(1), now(0)); // rotation angle of system1
        theta(1) = std::atan2(now(4), now(3)); // rotation angle of system2
        Eigen::VectorXd start_n = n; // preserve n at the start of the loop for stagger and step
         double max_duration; // max duration of laminar
        // calculate rotation number
        // no perturbation at first
        for (int j = 0; j < check_steps; j++){
            std::tie(n, theta, now) = calc_next(CR, n, theta, now);
            if (isLaminar(theta+n*2*M_PI, sync_criteria, d)){
                if (j < progress_steps){
                    now_time += dt;
                    trajectory.block(0, j+1, num_variables, 1) = now;
                    trajectory(num_variables, j+1) = now_time;
                }
                if (j+1 == progress_steps){
                    next_n = n; //preserve candidate of n
                }
            }
            else{
                laminar = false;
                max_duration = j*dt;
                break;
            }
        }
        // if laminar, continue to for loop
        if (laminar){
            CR.t_0 = trajectory.bottomRightCorner(1, 1)(0, 0);
            CR.x_0 = trajectory.topRightCorner(num_variables, 1);
            calced_laminar.middleCols(i*progress_steps, progress_steps+1) = trajectory;
            n = next_n;
            continue;
        }
        // otherwise, try stagger and step in parallel
        else{
            /*
             ███    ██    ███     ███ ██████   ██   ████  ██████
            █       ██   █       █       █     ██   █   █    █
            █      █ █   █       █       █    █ █   █   █    █
            ███    █  █  ███     ███     █    █  █  █   █    █
              ██  ██  █    ██      ██    █   ██  █  ████     █
               █  ██████    █       █    █   ██████ █  █     █
               █  █    █    █       █    █   █    █ █  ██    █
            ███  █     █ ███     ███     █  █     █ █   ██   █
            */
            std::cout << std::endl;
            int counter = 0;
            bool success = false; // whether stagger and step succeeded
            Eigen::VectorXd original_x_0 = CR.x_0; // log original x_0 for calc overall perturbation
            // parallelization doesn't work well without option
            #pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(success, max_duration, CR, n, counter, max_perturbation, original_x_0, numThreads) firstprivate(num_variables, sync_criteria, d, check_steps, progress_steps, start_n, perturb_min, perturb_max)
            for (int j = 0; j < limit; j++){
                if (success){
                    continue;
                }
                // show progress
                if (omp_get_thread_num() == 0)
                {
                    counter++;
                    std::cout << "\r " << counter * numThreads << "試行　最高" << max_duration << "/"<< check << std::flush;
                }
                bool Local_laminar = true; // flag
                CoupledRossler Local_CR = CR; // copy of CR
                double Local_now_time = Local_CR.t_0;
                Eigen::VectorXd Local_x_0 = myfunc::perturbation(Local_CR.x_0, perturb_min, perturb_max); // perturbed initial state
                Eigen::VectorXd Local_now = Local_x_0;
                Eigen::MatrixXd Local_trajectory = Eigen::MatrixXd::Zero(num_variables+1, progress_steps+1); //wide matrix for progress
                Local_trajectory.topLeftCorner(num_variables, 1) = Local_now;
                Local_trajectory(num_variables, 0) = Local_now_time;
                Eigen::VectorXd Local_theta(2);
                Local_theta(0) = std::atan2(Local_now(1), Local_now(0)); // rotation angle of system1
                Local_theta(1) = std::atan2(Local_now(4), Local_now(3)); // rotation angle of system2
                Eigen::VectorXd Local_n = start_n; // It doesn't matter if this is not accurate due to the perturbation, because wrong case won't survive.
                Eigen::VectorXd Local_next_n;
                for (int k = 0; k < check_steps; k++){
                    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> Local_next = calc_next(Local_CR, Local_n, Local_theta, Local_now);
                    Local_n = std::get<0>(Local_next);
                    Local_theta = std::get<1>(Local_next);
                    Local_now = std::get<2>(Local_next);
                    Local_now_time += Local_CR.dt;
                    if (isLaminar(Local_theta+Local_n*2*M_PI, sync_criteria, d)){
                        if (k < progress_steps){
                            Local_trajectory.block(0, k+1, num_variables, 1) = Local_now;
                            Local_trajectory(num_variables, k+1) = Local_now_time;
                        }
                        if (k+1 == progress_steps){
                            Local_next_n = Local_n; //preserve candidate of n
                        }
                    }
                    else{
                        #pragma omp critical
                        if (Local_now_time - Local_CR.t_0 > max_duration && success == false){
                            {
                                max_duration = Local_now_time - Local_CR.t_0;
                                // (CR.get_x_0_() - Local_x_0).norm(); //perturbation size
                                CR.x_0 = Local_x_0;
                            }
                        }
                        Local_laminar = false;
                        break;
                    }
                }
                /*
                 ████  █    █   ████   ████   ██████  ████   ████
                █  ██  █    █  ██  ██ ██  ██  █      █  ██  ██  ██
                █   █  █    █ ██    █ █    █  █      █   █  █    █
                ███    █    █ ██      █       █ ███  ███    ███
                  ███  █    █ ██      █       █ ███    ███    ███
                █   ██ █    █ ██    █ █    ██ █     ██   ██ █    █
                █   ██ ██  ██  █   ██ ██   █  █      █   ██ █   ██
                █████   ████   █████   ████   ██████ █████  █████
                */
                #pragma omp critical
                if (Local_laminar == true && success == false){
                    {   
                        double perturbation_size = (Local_trajectory.topLeftCorner(num_variables, 1) - original_x_0).norm();
                        if (perturbation_size > max_perturbation){
                            max_perturbation = perturbation_size;
                        }
                        if (perturbation_size < min_perturbation){
                            min_perturbation = perturbation_size;
                        }
                        std::cout << " overall perturbation scale here is " << perturbation_size << std::endl;
                        CR.t_0 = Local_trajectory.bottomRightCorner(1, 1)(0, 0);
                        CR.x_0 = Local_trajectory.topRightCorner(num_variables, 1);
                        calced_laminar.middleCols(i*progress_steps, progress_steps+1) = Local_trajectory;
                        n = Local_next_n;
                        success = true;
                    }
                }
            } // end of stagger and step for loop

            if (!success){
                std::cout << "stagger and step failed" << std::endl;
                // 成功した分だけcalced_laminarをresize
                calced_laminar.conservativeResize(num_variables+1, i*progress_steps+1);
                break;
            }
        }// end of stagger and step
    }

    // order of max perturbation
    std::cout << "max perturbation is " << max_perturbation << std::endl;
    std::cout << "min perturbation is " << min_perturbation << std::endl;
    int logged_max_perturbation = static_cast<int>(log10(max_perturbation)-0.5) - 1;
    std::cout << "logged max perturbation is " << logged_max_perturbation << std::endl;
    int logged_min_perturbation = static_cast<int>(log10(min_perturbation) - 0.5) -1;
    std::cout << "logged min perturbation is " << logged_min_perturbation << std::endl;

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
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 1200);
    std::vector<double> x(calced_laminar.cols()),y(calced_laminar.cols());
    int reach = static_cast<int>(calced_laminar.bottomRightCorner(1, 1)(0, 0) + 0.5); 

    for(int i=0;i<calced_laminar.cols();i++){
        x[i]=calced_laminar(0, i);
        y[i]=calced_laminar(3, i);
    }
    plt::plot(x,y);
    // 最後の点を赤でプロット(サイズは大きめ)
    std::vector<double> x_last(1), y_last(1);
    x_last[0] = x[x.size()-1];
    y_last[0] = y[y.size()-1];
    std::map<std::string, std::string> lastPointSettings;
    lastPointSettings["color"] = "red";
    lastPointSettings["marker"] = "o";
    lastPointSettings["markersize"] = "5";
    plt::plot(x_last, y_last, lastPointSettings);
    // 最初の点を緑でプロット(サイズは大きめ)
    std::vector<double> x_first(1), y_first(1);
    x_first[0] = x[0];
    y_first[0] = y[0];
    std::map<std::string, std::string> firstPointSettings;
    firstPointSettings["color"] = "green";
    firstPointSettings["marker"] = "o";
    firstPointSettings["markersize"] = "5";
    plt::plot(x_first, y_first, firstPointSettings);

    std::ostringstream oss;
    oss << "../../generated_lam_img/sync_gen_laminar_epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2  << "_" << check << "check" << progress << "progress10^" << logged_min_perturbation<<"-10^"<< logged_max_perturbation << "perturb.png";
    std::string filename = oss.str(); // 文字列を取得する
    if (calced_laminar.cols() > 1){
        std::cout << "\n Saving result to " << filename << std::endl;
        plt::save(filename);
    }

    /*
      ██████
     ███ ███
    ██                                             █
    ██                                             █
    ██          █████   ██      █   █████        ██████    █████          █ █████     █ █████   ██      █
     ██        ██  ███   █     ██  ██   ██         ██     ███  ██         ███  ███    ███  ███   █     ██
      ███           ██   ██    ██  █     █         █     ██     ██        ██    ██    ██    ██   ██    ██
        ███         ██   ██   ██  ██     ██        █     ██      █        █     ██    █      █   ██    █
          ██    ██████    █   ██  █████████        █     ██      █        █      █    █      ██   █   ██
           ██ ███   ██    ██  █   ██               █     ██      █        █      █    █      ██   ██  ██
           ██ ██    ██    ██ ██   ██               █     ██      █        █      █    █      █     █  █
           █  ██    ██     █ ██   ██               █      █     ██        █      █    ██    ██     █ ██
    ███  ███  ██   ███     ███     ███  ██         ██     ███  ██         █      █    ███  ███     ███
    ██████     ████  █      ██      ██████          ███    █████          █      █    █ █████       ██
                                                                                      █             ██
                                                                                      █             █
                                                                                      █            ██
                                                                                      █           ██
                                                                                      █         ███
    */

    oss.str("");
    if(calced_laminar.cols() > 1){
        if (progress == t){
            oss << "../../initials/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << ".npy";
            std::string fname = oss.str(); // 文字列を取得する
            std::cout << "saving as " << fname << std::endl;
            EigenVec2npy(calced_laminar.topLeftCorner(calced_laminar.rows()-1, 1).col(0), fname);
        } else{
            oss << "../../generated_lam/sync_gen_laminar_epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2  << "_" << check << "check" << progress << "progress10^" << logged_min_perturbation<<"-10^"<< logged_max_perturbation << "perturb.npy";
            std::string fname = oss.str(); // 文字列を取得する
            std::cout << "saving as " << fname << std::endl;
            EigenMat2npy(calced_laminar, fname);
        }
    }
    myfunc::duration(start);
}

bool isLaminar(Eigen::VectorXd phases, double sync_criteria, double center){
        bool is_sync = false;
        double phase_diff = std::abs(phases(0)-phases(1));
        if(center - sync_criteria <= phase_diff && phase_diff <= center + sync_criteria ||
            2*M_PI - center - sync_criteria <= phase_diff && phase_diff <= 2*M_PI - center + sync_criteria){
                is_sync = true;
        }
    return is_sync;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> calc_next(CoupledRossler& CR, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXd previous){
    Eigen::VectorXd now = CR.rk4(previous);
    Eigen::VectorXd theta(2);
    theta(0) = std::atan2(now(1), now(0)); // rotation angle of system1
    theta(1) = std::atan2(now(4), now(3)); // rotation angle of system2
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = myfunc::shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}