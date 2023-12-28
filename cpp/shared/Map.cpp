/**
 * @file Map.cpp
 * @author Hibiki Kato
 * @brief Map classes(including poincare(&lorenz) map)
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "Map.hpp"
#include <eigen3/Eigen/Dense>
#include <set>
#include <vector>
#include <cmath>

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

PoincareMap::PoincareMap(const Eigen::MatrixXd& input_trajectory){
    trajectory = input_trajectory;
    std::vector<std::set<long long>> indices;
}

PoincareMap::~PoincareMap(){
}

void PoincareMap::locmax(int dim){
    std::set<long long> index;
    double diff1, diff2, diff3, diff4, diff5, diff6;
    for (long long i = 3; i < trajectory.cols()-3; ++i){
        //キャッシュを効かせるために変数に引き算を格納
        diff1 = trajectory(dim, i-2) - trajectory(dim, i-3);
        diff2 = trajectory(dim, i-1) - trajectory(dim, i-2);
        diff3 = trajectory(dim, i) - trajectory(dim, i-1);
        diff4 = trajectory(dim, i+1) - trajectory(dim, i);
        diff5 = trajectory(dim, i+2) - trajectory(dim, i+1);
        diff6 = trajectory(dim, i+3) - trajectory(dim, i+2);
        // 比較演算の回数を減らすために外側から多重if文を使う
        if (diff1 > 0){
            if (diff6 < 0){
                if (diff2 > 0){
                    if (diff5 < 0){
                        if (diff3 > 0){
                            if (diff4 < 0){
                                index.insert(i);
                            }
                        }
                    }
                }
            }
        }//if終わり
    }
    indices.push_back(index);
}

void PoincareMap::locmin(int dim){
    std::set<long long> index;
    double diff1, diff2, diff3, diff4, diff5, diff6;
    for (long long i = 3; i < trajectory.cols()-3; ++i){
        //キャッシュを効かせるために変数に引き算を格納
        diff1 = trajectory(dim, i-2) - trajectory(dim, i-3);
        diff2 = trajectory(dim, i-1) - trajectory(dim, i-2);
        diff3 = trajectory(dim, i) - trajectory(dim, i-1);
        diff4 = trajectory(dim, i+1) - trajectory(dim, i);
        diff5 = trajectory(dim, i+2) - trajectory(dim, i+1);
        diff6 = trajectory(dim, i+3) - trajectory(dim, i+2);
        // 比較演算の回数を減らすために外側から多重if文を使う
        if (diff1 < 0){
            if (diff6 > 0){
                if (diff2 < 0){
                    if (diff5 > 0){
                        if (diff3 < 0){
                            if (diff4 > 0){
                                index.insert(i);
                            }
                        }
                    }
                }
            }
        }//if終わり
    }
    indices.push_back(index);
}

void PoincareMap::poincare_section(int dim, double value){
    std::set<long long> index;
    double diff1, diff2;
    for (long long i = 0; i < trajectory.cols()-1; ++i){
        diff1 = trajectory(dim, i) - value;
        diff2 = trajectory(dim, i+1) - value;
        if (diff1 <= 0){
            if (diff2 >= 0){
                index.insert(i);
                index.insert(i+1);
            }
        }else{
            if (diff2 <= 0){
                index.insert(i);
                index.insert(i+1);
            }
        }
    }
    indices.push_back(index);
}

Eigen::MatrixXd PoincareMap::get(){
    //indiciesの中のsetを全て結合する
    std::set<long long> index = indices[0];
    for (int i = 1; i < indices.size(); ++i){
        index.merge(indices[i]);
    }
    auto itr = index.begin();
    //setの要素数
    long long size = index.size();
    Eigen::MatrixXd output(trajectory.rows(), size);
    //setの要素のインデックスをoutputに順次格納
    for (int i = 0; i < size; ++i){
        output.col(i) = trajectory.col(*itr);
        itr++;
    }
    return output;
}



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

KostelichMap::KostelichMap(KMparams input_params, long long input_n, long long input_dump, Eigen::VectorXd input_x_0){
    alpha = input_params.alpha;
    sigma = input_params.sigma;
    n = input_n;
    dump = input_dump;
    x_0 = input_x_0;
}
KostelichMap::~KostelichMap(){}

Eigen::MatrixXd KostelichMap::get_trajectory(){
    Eigen::MatrixXd trajectory(2, n+1);
    trajectory.col(0) = x_0;
    for (long long i = 0; i < dump; ++i){
        trajectory.col(0) = map(trajectory.col(0));
    }
    for (long long i = 0; i < n; ++i){
        trajectory.col(i+1) = map(trajectory.col(i));
    }
    return trajectory;
}

Eigen::VectorXd KostelichMap::map(const Eigen::VectorXd& state){
    Eigen::VectorXd output(2);
    output(0) = 3 * state(0);
    output(0) -= std::floor(output(0));

    output(1) = state(1) - sigma*std::sin(2*M_PI*state(1)) + alpha * (1 - std::cos(2*M_PI*state(0)));
    output(1) -= std::floor(output(1));
    return output;
}

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

PGMap::PGMap(PGparams input_params, long long input_n, long long input_dump, Eigen::VectorXd input_x_0){
    a = input_params.a;
    a_prime = input_params.a_prime;
    omega = input_params.omega;
    n = input_n;
    dump = input_dump;
    x_0 = input_x_0;
}
PGMap::~PGMap(){}
Eigen::MatrixXd PGMap::get_trajectory(){
    Eigen::MatrixXd trajectory(2, n+1);
    trajectory.col(0) = x_0;
    for (long long i = 0; i < dump; ++i){
        trajectory.col(0) = map(trajectory.col(0));
    }
    for (long long i = 0; i < n; ++i){
        trajectory.col(i+1) = map(trajectory.col(i));
    }
    return trajectory;

}
Eigen::VectorXd PGMap::map(const Eigen::VectorXd& state){
    Eigen::VectorXd output(2);
    output(0) = (1-omega) * tent_map(state(0), a) + omega * tent_map(state(1), a_prime);
    output(1) = omega * tent_map(state(0), a) + (1-omega) * tent_map(state(1), a_prime);
    return output;
}
double PGMap::tent_map(double x, double r){
    if (x < 1 / r){
        return r * x;
    }
    else{
        return r / (1 - r) * (x - 1);
    }
}