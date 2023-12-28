/**
 * @file Flow.hpp
 * @author Hibiki Kato
 * @brief header of Flow classes
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <complex>

/*
  █████    ██                  ██   ██   ███        ███                     █             ██
 ████████  ██                  ██   ██   ████       ███                     █             ██
██     ██  ██                  ██   ██   ████       ███                     █             ██
██      ██ ██                  ██   ██   ████      ████                     █             ██
██         ██ ████     ████    ██   ██   ██ ██     ████     █████      ██████     ████    ██
 ██        ████████   ███████  ██   ██   ██ ██     █ ██    ███████    ███████    ███████  ██
  ████     ██    ██  ██    ██  ██   ██   ██  █    ██ ██   ██    ███  ██    ██   ██    ██  ██
    ████   ██    ██  ██     █  ██   ██   ██  ██   ██ ██   ██     ██  ██     █   ██     █  ██
      ███  ██    ██  ████████  ██   ██   ██  ██  ██  ██   ██     ██  ██     █   ████████  ██
       ███ ██    ██  ██        ██   ██   ██   ██ ██  ██   ██     ██  ██     █   ██        ██
██      ██ ██    ██  ██        ██   ██   ██   ██ █   ██   ██     ██  ██     █   ██        ██
██     ██  ██    ██  ██        ██   ██   ██    ███   ██   ██    ███  ██    ██   ██        ██
 ████████  ██    ██   ███████  ██   ██   ██    ███   ██    ███████    ███████    ███████  ██
  █████    ██    ██    █████   ██   ██   ██    ██    ██     █████      ████ █     █████   ██
*/

struct SMparams{
    double nu;
    double beta;
    std::complex<double> f;
};

struct ShellModel{   
    //constructor
    ShellModel(SMparams input_prams, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXcd input_x_0);
    ~ShellModel();
    Eigen::MatrixXcd get_trajectory();
    Eigen::VectorXd energy_spectrum(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
    Eigen::VectorXcd rk4(const Eigen::VectorXcd& present);
    Eigen::VectorXcd goy_shell_model(const Eigen::VectorXcd& state);
    Eigen::MatrixXd jacobian_matrix(const Eigen::VectorXd& state);
    void set_beta_(double input_beta);
    void set_t_0_(double input_t_0);
    void set_t_(double input_t);

    //data members
    double nu;
    double beta;
    std::complex<double> f;
    double dt;
    double t_0;
    double t;
    double dump;
    long steps;
    long dump_steps;
    Eigen::VectorXd k_n;
    Eigen::VectorXd c_n_1;
    Eigen::VectorXd c_n_2;
    Eigen::VectorXd c_n_3;
    Eigen::VectorXcd x_0;
};

/*
   █████                                    ██                    █   ████████                                   ██
  ███████                                   ██                    █   █████████                                  ██
 ██     ██                                  ██                    █   ██     ███                                 ██
 ██     ██                                  ██                    █   ██      ██                                 ██
██       █    █████    ██    ██  ██ ████    ██     ████      ██████   ██      ██    █████      ████      ████    ██     ████    ██ ██
██           ███████   ██    ██  ████████   ██    ███████   ███████   ██      ██   ███████    ██████    ██████   ██    ███████  █████
██          ██    ███  ██    ██  ██    ██   ██   ██    ██  ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██    ██  ██
██          ██     ██  ██    ██  ██     ██  ██   ██     █  ██     █   ████████    ██     ██  ██        ██        ██   ██     █  ██
██          ██     ██  ██    ██  ██     ██  ██   ████████  ██     █   ███████     ██     ██   ████      ████     ██   ████████  ██
██       █  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██    ██    ██     ██     ████      ████   ██   ██        ██
 ██     ██  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██     ██   ██     ██        ██        ██  ██   ██        ██
 ██     ██  ██    ███  ██    ██  ██    ██   ██   ██        ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██        ██
  ███████    ███████    ███████  ████████   ██    ███████   ███████   ██      ██   ███████   ███████   ███████   ██    ███████  ██
   █████      █████      ██████  ██ ████    ██     █████     ████ █   ██      ██    █████      ████      ████    ██     █████   ██
                                 ██
                                 ██
                                 ██
                                 ██
*/

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
    long long steps;
    long long dump_steps;
};

namespace myfunc{
    // Jacobian matrix
    Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt);
    Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian);
    Eigen::MatrixXd regularizeJacobian(const Eigen::MatrixXd& jacobian);

    // phase synchronization
    double shift(double pre_theta, double theta, double rotation_number);
    bool isSync(double a, double b, double sync_criteria, double center = 0);
}