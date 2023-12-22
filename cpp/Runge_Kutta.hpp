#pragma once
#include <eigen3/Eigen/Dense>

struct CRparams{
    double omega1;
    double omega2;
    double epsilon;
    double a;
    double c;
    double f;
};

struct CoupledRossler{
    CoupledRossler(CRparams input_params, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0);
    ~CoupledRossler();
    Eigen::MatrixXd get_trajectory();
    Eigen::VectorXd rk4(const Eigen::VectorXd& present);
    Eigen::VectorXd coupled_rossler(const Eigen::VectorXd& present);
    Eigen::MatrixXd jacobi_matrix(const Eigen::VectorXd& state);
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




