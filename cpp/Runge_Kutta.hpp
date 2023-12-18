#pragma once
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <iostream>
#include <random>
#include <omp.h>

struct CoupledRossler{
    CoupledRossler(double input_omega1, double input_omega2, double input_epsilon, double input_a, double input_c, double input_f, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0);
    ~CoupledRossler();
    Eigen::MatrixXd get_trajectory_();
    Eigen::VectorXd rk4_(const Eigen::VectorXd& present);
    Eigen::VectorXd coupledRossler(const Eigen::VectorXd& present);
    double omega1;
    double omega2;
    double epsilon;
    double a;
    double c;
    double f;
    double dt;
    double t_0;
    double t;
    double dump;
    Eigen::VectorXd x_0;
    long steps;
    long dump_steps;
};




