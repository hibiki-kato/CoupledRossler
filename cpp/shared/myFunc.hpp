/**
 * @file myFunc.hpp
 * @author Hibiki Kato
 * @brief header of my functions
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <chrono>
#include <random>
#include <eigen3/Eigen/Dense>

namespace myfunc{
    std::string ordinal_suffix(int n);
    void duration(std::chrono::time_point<std::chrono::system_clock> start);

    template <typename Vector>
    Vector multi_scale_perturbation(Vector state, int s_min, int s_max){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s(-1, 1);
        std::uniform_real_distribution<double> expo(s_min, s_max);
        Vector uniformed(state.rows());
        for(int i = 0; i < state.rows(); i++){
            uniformed(i) = s(gen);
        }
        Vector unit = state.cwiseProduct(uniformed);
        unit /= unit.norm();

        return (unit.array() * std::pow(10, expo(gen)) + state.array()).matrix();
    }
    template <typename Vector>
    Vector perturbation(Vector state, int s_min, int s_max){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s(-1, 1);
        std::uniform_real_distribution<double> expo(s_min, s_max);

        Vector unit(state.rows());
        for(int i = 0; i < state.rows(); i++){
            unit(i) = s(gen);
        }
        unit /= unit.norm();
        return (unit.array() * std::pow(10, expo(gen)) + state.array()).matrix();
    }
}
