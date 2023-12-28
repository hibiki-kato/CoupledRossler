/**
 * @file Map.hpp
 * @author Hibiki Kato
 * @brief header of Maps(including poincare(&lorenz) map)
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * )
 */
#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "myFunc.hpp"
#include <set>
#include <vector>


/*
████████               ██                                                 ███        ███
█████████              ██                                                 ████       ███
██      ██                                                                ████       ███
██      ██                                                                ████      ████
██      ██    █████    ██   ██ ████     ████      ████    ██ ██   ████    ██ ██     ████     ████    ██ ████
██      ██   ███████   ██   ████████   ███████   ███ ██   █████  ███████  ██ ██     █ ██    ███ ██   ████████
██     ███  ██    ███  ██   ██    ██  ██    ██  ██    ██  ██    ██    ██  ██  █    ██ ██   ██    ██  ██    ██
█████████   ██     ██  ██   ██    ██  ██     █        ██  ██    ██     █  ██  ██   ██ ██         ██  ██     ██
██████      ██     ██  ██   ██    ██  ██          ██████  ██    ████████  ██  ██  ██  ██     ██████  ██     ██
██          ██     ██  ██   ██    ██  ██        ███   ██  ██    ██        ██   ██ ██  ██   ███   ██  ██     ██
██          ██     ██  ██   ██    ██  ██        ██    ██  ██    ██        ██   ██ █   ██   ██    ██  ██     ██
██          ██    ███  ██   ██    ██  ██    ██  ██    ██  ██    ██        ██    ███   ██   ██    ██  ██    ██
██           ███████   ██   ██    ██   ███████  ████████  ██     ███████  ██    ███   ██   ████████  ████████
██            █████    ██   ██    ██    ████     ████ ██  ██      █████   ██    ██    ██    ████ ██  ██ ████
                                                                                                     ██
                                                                                                     ██
                                                                                                     ██
                                                                                                     ██
*/

struct PoincareMap{
    PoincareMap(const Eigen::MatrixXd& input_trajectory);
    ~PoincareMap();
    void locmax(int dim);
    void locmin(int dim);
    void poincare_section(int dim, double value);
    Eigen::MatrixXd get();
    Eigen::MatrixXd trajectory;
    std::vector<std::set<long long>> indices;
};

/*
██      ██                                       ██   ██             ██        ███        ███
██     ██                                        ██   ██             ██        ████       ███
██    ██                          ██             ██                  ██        ████       ███
██   ███                          ██             ██                  ██        ████      ████
██   ██       █████      ████   ██████   ████    ██   ██     ████    ██ ████   ██ ██     ████     ████    ██ ████
██  ██       ███████    ██████    ██    ███████  ██   ██    ███████  ████████  ██ ██     █ ██    ███ ██   ████████
██████      ██    ███  ██    ██   ██   ██    ██  ██   ██   ██    ██  ██    ██  ██  █    ██ ██   ██    ██  ██    ██
██████      ██     ██  ██         ██   ██     █  ██   ██   ██     █  ██    ██  ██  ██   ██ ██         ██  ██     ██
███  ██     ██     ██   ████      ██   ████████  ██   ██   ██        ██    ██  ██  ██  ██  ██     ██████  ██     ██
██    ██    ██     ██     ████    ██   ██        ██   ██   ██        ██    ██  ██   ██ ██  ██   ███   ██  ██     ██
██    ██    ██     ██        ██   ██   ██        ██   ██   ██        ██    ██  ██   ██ █   ██   ██    ██  ██     ██
██     ██   ██    ███  ██    ██   ██   ██        ██   ██   ██    ██  ██    ██  ██    ███   ██   ██    ██  ██    ██
██      ██   ███████   ███████    ███   ███████  ██   ██    ███████  ██    ██  ██    ███   ██   ████████  ████████
██      ███   █████      ████      ███   █████   ██   ██     ████    ██    ██  ██    ██    ██    ████ ██  ██ ████
                                                                                                          ██
                                                                                                          ██
                                                                                                          ██
                                                                                                          ██
*/
struct KMparams{
    double alpha;
    double sigma;
};

struct KostelichMap{
    KostelichMap(KMparams input_params, long long input_n, long long input_dump, Eigen::VectorXd input_x_0);
    ~KostelichMap();
    Eigen::MatrixXd get_trajectory();
    Eigen::VectorXd map(const Eigen::VectorXd& state);
    Eigen::MatrixXd jacobi_matrix(const Eigen::VectorXd& state);
    double alpha;
    double sigma;
    long long n;
    long long dump;
    Eigen::VectorXd x_0;
};

/*
████████    ██   ██                                      ██                         █████                                         ██                                                    ███        ███
█████████   ██   ██                                      ██                        ███████                                        ██                                                    ████       ███
██      ██       ██                                      ██                       ██     ██                                       ██                                                    ████       ███
██      ██       ██                                      ██                       ██     ██                                       ██                                                    ████      ████
██      ██  ██   ██    ██    █████   ██     ██   ████    ██    ██ ██     ██      ██           ██ ██   ████      ████      ████    ██ ████      ████    ██ ██   ████ █     ████    ██ ██ ██ ██     ████     ████    ██ ████
██      ██  ██   ██   ██    ███████   ██    █   ██████   ██   ██   ██   ██       ██           █████  ███ ██    ██████    ██████   ████████    ███████  █████  ███████    ███████  █████ ██ ██     █ ██    ███ ██   ████████
██     ███  ██   ██  ██    ██    ███  ██   ██  ██    ██  ██  ██    ██   ██       ██           ██    ██    ██  ██    ██  ██    ██  ██    ██   ██    ██  ██    ██    ██   ██    ██  ██    ██  █    ██ ██   ██    ██  ██    ██
█████████   ██   ██ ██     ██     ██   █   ██  ██        ██ ██     ██   ██       ██    █████  ██          ██  ██        ██        ██     ██  ██     █  ██    ██     █   ██     █  ██    ██  ██   ██ ██         ██  ██     ██
██████      ██   █████     ██     ██   ██  █    ████     █████      ██ ██  █████ ██    █████  ██      ██████   ████      ████     ██     ██  ████████  ██    ██     █   ████████  ██    ██  ██  ██  ██     ██████  ██     ██
██          ██   █████     ██     ██   ██ ██      ████   █████      ██ ██        ██       ██  ██    ███   ██     ████      ████   ██     ██  ██        ██    ██     █   ██        ██    ██   ██ ██  ██   ███   ██  ██     ██
██          ██   ██  ██    ██     ██    █ ██         ██  ██  ██     ██ ██         ██      ██  ██    ██    ██        ██        ██  ██     ██  ██        ██    ██     █   ██        ██    ██   ██ █   ██   ██    ██  ██     ██
██          ██   ██   ██   ██    ███    ███    ██    ██  ██   ██     ███          ██      ██  ██    ██    ██  ██    ██  ██    ██  ██    ██   ██        ██    ██    ██   ██        ██    ██    ███   ██   ██    ██  ██    ██
██          ██   ██   ███   ███████     ███    ███████   ██   ███    ███           ████████   ██    ████████  ███████   ███████   ████████    ███████  ██     ███████    ███████  ██    ██    ███   ██   ████████  ████████
██          ██   ██    ██    █████       █       ████    ██    ██     █             ██████    ██     ████ ██    ████      ████    ██ ████      █████   ██      ██████     █████   ██    ██    ██    ██    ████ ██  ██ ████
                                                                     ██                                                                                            ██                                              ██
                                                                     ██                                                                                       █    ██                                              ██
                                                                   ███                                                                                        ███████                                              ██
                                                                   ██                                                                                          ████                                                ██
*/

struct PGparams{
    double a;
    double a_prime;
    double omega;
};

struct PGMap{
    PGMap(PGparams input_params, long long input_n, long long input_dump, Eigen::VectorXd input_x_0);
    ~PGMap();
    Eigen::MatrixXd get_trajectory();
    Eigen::VectorXd map(const Eigen::VectorXd& state);
    double tent_map(const double x, double r);
    Eigen::MatrixXd jacobi_matrix(const Eigen::VectorXd& state);
    double a;
    double a_prime;
    double omega;
    long long n;
    long long dump;
    Eigen::VectorXd x_0;
};

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
namespace myfunc{
    template <typename MapObj>
    Eigen::MatrixXd SaS_of_map(MapObj MO, std::function<bool(Eigen::VectorXd, double)> isLaminar, double epsilon, int progress, int check, double perturb_min, double perturb_max, int limit, int numThreads){
        long long num_variables = MO.x_0.size();
        Eigen::MatrixXd calced_laminar(num_variables, MO.n+1);
        int stagger_and_step_num = static_cast<int>(MO.n / progress + 0.5);
        MO.n = check; // for what??

        double max_perturbation = 0; // max perturbation
        double min_perturbation = 1; // min perturbation

        for (int i=0; i < stagger_and_step_num; i++){
            std::cout << "\r 現在" << i * progress << "時間" << std::flush;
            bool laminar = true; // flag
            
            Eigen::VectorXd now = MO.x_0; // initial state
            Eigen::MatrixXd trajectory = Eigen::MatrixXd::Zero(num_variables, progress+1); //wide matrix for progress
            trajectory.col(0) = now;
            double max_duration; // max duration of laminar
            // no perturbation at first
            for (int j = 0; j < check; j++){
                now = MO.map(now);
                if (isLaminar(now, epsilon)){
                    if (j < progress){
                        trajectory.col(j+1) = now;
                    }
                }
                else{
                    laminar = false;
                    max_duration = j;
                    break;
                }
            }
            // if laminar, continue to for loop
            if (laminar){
                MO.x_0 = trajectory.col(progress);
                calced_laminar.middleCols(i*progress, progress+1) = trajectory;
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
                Eigen::VectorXd original_x_0 = MO.x_0; // log original x_0 for calc overall perturbation
                // parallelization doesn't work well without option
                #pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(success, max_duration, MO, counter, max_perturbation, min_perturbation, original_x_0, numThreads) firstprivate(num_variables, check, progress, perturb_min, perturb_max)
                for (int j = 0; j < limit; j++){
                    if (success){
                        continue;
                    }
                    // if (omp_get_thread_num() == 0){
                    //     if (j%100000 == 0) std::cout << "\r " << counter << "試行　最高" << max_duration << "/"<< check << std::flush;
                    // }

                    bool Local_laminar = true; // flag
                    MapObj Local_MO = MO; // copy of MO
                    Eigen::VectorXd Local_x_0 = myfunc::perturbation(Local_MO.x_0, perturb_min, perturb_max); // perturbed initial state
                    Eigen::VectorXd Local_now = Local_x_0;
                    Eigen::MatrixXd Local_trajectory = Eigen::MatrixXd::Zero(num_variables, progress+1); //wide matrix for progress
                    Local_trajectory.col(0) = Local_now;
                    for (int k = 0; k < check; k++){
                        Local_now = MO.map(Local_now);
                        if (isLaminar(Local_now, epsilon)){
                            if (k < progress){
                                Local_trajectory.col(k+1) = Local_now;
                            }
                        }
                        else{
                            #pragma omp critical
                            if (k > max_duration && success == false){
                                {
                                    max_duration =  k;
                                    MO.x_0 = Local_x_0;
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
                            double perturbation_size = (Local_trajectory.col(0) - original_x_0).norm();
                            if (perturbation_size > max_perturbation){
                                max_perturbation = perturbation_size;
                            }
                            if (perturbation_size < min_perturbation){
                                min_perturbation = perturbation_size;
                            }
                            std::cout << counter << "trial. overall perturbation scale here is " << perturbation_size << std::endl;
                            MO.x_0 = Local_trajectory.col(progress);
                            calced_laminar.middleCols(i*progress, progress+1) = Local_trajectory;
                            success = true;
                        }
                    }
                    #pragma omp atomic
                    counter++;
                } // end of stagger and step for loop

                if (!success){
                    std::cout << "stagger and step failed" << std::endl;
                    // 成功した分だけcalced_laminarをresize
                    calced_laminar.conservativeResize(num_variables, i*progress+1);
                    break;
                }
            }// end of stagger and step
        }// end of calc

        return calced_laminar;
    }
}